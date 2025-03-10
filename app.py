#! /usr/bin/env python

"""nlpdb."""

import io
import logging
import re
from functools import wraps
from os import getenv
from pathlib import Path

import mysql.connector
import pandas as pd
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    session,
)
from openai import OpenAI
from rich import box
from rich.console import Console
from rich.table import Table

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


load_dotenv()
app = Flask(__name__)
app.secret_key = getenv("FLASK_SECRET_KEY")
admin_username = getenv("ADMIN_USERNAME")
admin_password = getenv("ADMIN_PASSWORD")

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


class ValidationError(Exception):
    """Custom validation error class."""

    def __init__(
        self, field: str, message: str = "A validation error occurred"
    ) -> None:
        """Handle message."""
        self.field = field
        self.message = message
        super().__init__(f"{message}: {field}")


class ChatGPT:
    """ChatGPT Language model handler."""

    client: OpenAI = None

    def __init__(self) -> None:
        """Validate requirements and set constants."""
        api_key = getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str, model: str = "gpt-4") -> str:
        """Query ChatGPT with a prompt.

        Args:
            prompt: The text prompt to send to ChatGPT
            model: The OpenAI model to use (defaults to gpt-4)

        Returns:
            The model's response text

        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content


class MySQLDB:
    """MySQL database connection and query handler."""

    def __init__(self) -> None:
        """Initialize database connection using environment variables."""
        self.host = getenv("MYSQL_HOST")
        self.user = getenv("MYSQL_USER")
        self.password = getenv("MYSQL_PASSWORD")
        self.database = getenv("MYSQL_DATABASE")
        self.connection = None

    def _validate_table_name(self, table_name: str) -> None:
        """Validate table name contains only allowed characters."""
        if not table_name.isalnum() and not all(
            c.isalnum() or c == "_" for c in table_name
        ):
            err = "Table name must be only alphanumeric characters or underscores"
            raise ValidationError(err)

    def connect(self) -> None:
        """Establish database connection."""
        self.connection = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database,
        )

    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def execute_query(self, query: str, params: tuple | None = None) -> pd.DataFrame:
        """Execute a SQL query and return results as a pandas DataFrame.

        Args:
            query: SQL query string
            params: Optional tuple of parameters for parameterized queries

        Returns:
            pandas DataFrame containing query results

        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)

            if cursor.description:
                results = cursor.fetchall()
                return pd.DataFrame(results)

            self.connection.commit()
            return pd.DataFrame({"affected_rows": [cursor.rowcount]})
        finally:
            if cursor:
                cursor.close()

    def import_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Import a pandas DataFrame into a MySQL table."""
        cursor = None
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()

            df = df.replace({float("nan"): None})
            df = df.where(pd.notna(df), None)

            cursor = self.connection.cursor()
            cursor.execute("SET SESSION net_read_timeout=3600")
            cursor.execute("SET SESSION net_write_timeout=3600")
            cursor.execute("SET SESSION interactive_timeout=28800")
            cursor.execute("SET SESSION wait_timeout=28800")
            cursor.close()
            cursor = None

            self._validate_table_name(table_name)

            columns = []
            for col, dtype in df.dtypes.items():
                if "int" in str(dtype):
                    sql_type = "INT NULL"
                elif "float" in str(dtype):
                    sql_type = "FLOAT NULL"
                else:
                    sql_type = "TEXT NULL"
                col_quoted = f"`{col.replace('`', '``')}`"
                columns.append(f"{col_quoted} {sql_type}")

            table_quoted = f"`{table_name.replace('`', '``')}`"
            create_table = (
                f"CREATE TABLE IF NOT EXISTS {table_quoted} ({', '.join(columns)})"
            )

            logger.info("Creating table with query: %s", create_table)
            self.execute_query(create_table)

            values = (
                df.replace({pd.NA: None, float("nan"): None})
                .to_records(index=False)
                .tolist()
            )

            batch_size = 5000
            cursor = self.connection.cursor()

            for i in range(0, len(values), batch_size):
                batch = values[i : i + batch_size]
                logger.info(
                    "Inserting batch %d of %d",
                    i // batch_size + 1,
                    len(values) // batch_size + 1,
                )
                cursor.executemany(
                    f"INSERT INTO {table_quoted} VALUES ({','.join(['%s'] * len(batch[0]))})",  # noqa: S608, E501
                    batch,
                )
                self.connection.commit()

        except Exception:
            logger.exception("Error importing dataframe")
            if cursor:
                cursor.close()
            self.disconnect()
            raise

        finally:
            if cursor:
                cursor.close()
            self.disconnect()


class CommandProcessor:
    """Terminal configuration and behavior."""

    def __init__(self) -> None:
        """Initialize the command processor."""
        self.current_df = None
        self.chatgpt = ChatGPT()
        self.mysql = MySQLDB()
        self.commands = {
            "help": lambda _: Path("templates/help.menu").read_text(),
            "import": lambda _: "OPEN_FILE_DIALOG",
            "clear": lambda _: "CLEAR_SCREEN",
        }

    def _get_sql_query(self, command: str) -> str:
        """Get SQL query from ChatGPT."""
        command = command.replace("= '", "='").replace("='", "= '").replace("'", "'")
        query_filters = [
            "return only a MySQL query",
            "do not include any backticks",
            "do not include any markdown",
            "do not include any explanation",
            "only return the raw SQL query",
        ]
        prompt = " ".join(["Write me a MySQL query to ", command, *query_filters])

        sql_query = self.chatgpt.query(prompt).strip()
        for prefix in ["```sql", "```"]:
            if sql_query.startswith(prefix):
                sql_query = sql_query.removeprefix(prefix)
        return sql_query.removesuffix("```").strip()

    def _format_results(self, result: pd.DataFrame) -> str:
        """Format SQL results into a rich table output."""
        if not isinstance(result, pd.DataFrame) or result.empty:
            return "Query returned no results"

        console = Console(
            file=io.StringIO(), force_terminal=True, width=120, color_system="standard"
        )

        table = Table(
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold white",
            show_lines=True,
            show_edge=True,
            padding=(0, 1),
            border_style="white",
        )

        num_columns = len(result.columns)
        available_width = console.width - (num_columns * 3) - 2
        min_width = max(10, available_width // (num_columns * 2))
        max_width = max(20, available_width // num_columns)

        for col in result.columns:
            table.add_column(
                str(col),
                justify="left",
                style="white",
                min_width=min_width,
                max_width=max_width,
                overflow="fold",
            )

        for _, row in result.iterrows():
            formatted_values = [
                str(val) if pd.notna(val) and val is not None else "" for val in row
            ]
            table.add_row(*formatted_values)

        console.print(table)
        output = console.file.getvalue()

        return re.sub(r"\x1b\[[0-9;]*[mGKH]", "", output)

    def process_command(self, command: str) -> str:
        """Process commandline."""
        command = command.strip()
        if not command:
            return None

        parts = command.split()
        cmd = parts[0].lower()

        if cmd in self.commands:
            result = self.commands[cmd](parts[1:] if len(parts) > 1 else [])
            return "OPEN_FILE_DIALOG_NO_PROMPT" if cmd == "import" else result

        try:
            sql_query = self._get_sql_query(command)
            logger.info("Executing query: %s", sql_query)
            query_message = f"Executing query: {sql_query}\n\n"
            result = self.mysql.execute_query(sql_query)
            formatted_result = self._format_results(result)
            return query_message + formatted_result
        except Exception as e:
            return f"SQL Error: {e}"

    def process_csv(self, file_content: bytes, filename: str) -> str:
        """Process uploaded CSV file and import to database.

        Args:
            file_content: Raw bytes of the CSV file
            filename: Original filename of the uploaded file

        Returns:
            Status message

        """
        try:
            csv_file = io.BytesIO(file_content)
            df = pd.read_csv(csv_file)

            if df.empty:
                return "Error: The CSV file is empty"

            table_name = Path(filename).stem.lower()

            logger.info("Starting import of %d rows to table '%s'", len(df), table_name)

            self.mysql.import_dataframe(df, table_name)

            return f"Successfully imported {len(df)} rows to table '{table_name}'"
        except pd.errors.EmptyDataError:
            return "Error: The CSV file is empty"
        except pd.errors.ParserError as e:
            return f"Error: Failed to parse CSV file: {e}"
        except Exception as e:
            logger.exception("CSV Import Error")
            return f"Error importing CSV: {e}"


processor = CommandProcessor()


def login_required(f: callable) -> callable:
    """Provide basic authentication."""

    @wraps(f)
    def decorated_function(*args: tuple, **kwargs: dict) -> str:
        if "authenticated" not in session:
            return render_template("login.html")
        return f(*args, **kwargs)

    return decorated_function


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    """Process login attempt."""
    if request.method == "GET":
        if "authenticated" in session:
            return redirect("/")
        return render_template("login.html")

    logger.info("Login attempt for user: %s", request.form.get("username"))

    if (
        request.form["username"] == admin_username
        and request.form["password"] == admin_password
    ):
        session["authenticated"] = True
        logger.info("Login successful")
        return redirect("/")

    logger.info("Login failed")
    return render_template("login.html", error="Invalid credentials")


@app.route("/", methods=["GET", "POST"])
@login_required
def home() -> str:
    """Render homescreen."""
    return render_template("terminal.html")


@app.route("/execute", methods=["POST"])
def execute() -> Response:
    """Command handler."""
    command = request.json.get("command", "")
    if not command:
        return jsonify({"error": "No command provided"}), 400

    try:
        output = processor.process_command(command)
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload_csv", methods=["POST"])
def upload_csv() -> Response:
    """CSV handler."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        file_content = file.read()
        result = processor.process_csv(file_content, file.filename)
        return jsonify({"output": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    options = {
        "host": str(getenv("FLASK_HOST", "127.0.0.1")),
        "port": int(getenv("FLASK_PORT", "8080")),
        "debug": bool(getenv("FLASK_DEBUG", "False")),
    }

    app.run(**options)
