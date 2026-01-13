from pathlib import Path

FILES = [
    "ascii_dalton.txt",
    "ascii_dirac.txt",
    "ascii_schrodinger.txt",
    "ascii_bohr.txt",
    "ascii_rutherford.txt",
]

OUTPUT = "ascii_data.py"


def escape_triple_quotes(text: str) -> str:
    """Escape triple quotes so we can safely embed text in ''' ... '''"""
    return text.replace("'''", "\\'\\'\\'")


def main():
    data = {}

    for fname in FILES:
        path = Path(fname)
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {fname}")

        key = path.stem.replace("ascii_", "")
        text = path.read_text(encoding="utf-8")
        data[key] = escape_triple_quotes(text)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("# Auto-generated file. Do not edit manually.\n\n")
        f.write("ASCII_ART = {\n")

        for key, text in data.items():
            f.write(f"    '{key}': '''\n{text}\n''',\n\n")

        f.write("}\n")

    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
