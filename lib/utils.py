"""Small notebook display helpers."""

from html import escape


def table(headers, rows):
    """Display a bounded table, keeping dataset text as text."""
    from IPython.display import HTML, display

    def row(values, tag):
        return (
            "<tr>"
            + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values)
            + "</tr>"
        )

    display(
        HTML(
            "<table><thead>"
            + row(headers, "th")
            + "</thead><tbody>"
            + "".join(row(values, "td") for values in rows)
            + "</tbody></table>"
        )
    )
