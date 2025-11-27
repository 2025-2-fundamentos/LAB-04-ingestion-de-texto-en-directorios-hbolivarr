# pylint: disable=import-outside-toplevel
# pylint: disable=line-too-long
# flake8: noqa
"""
Escriba el codigo que ejecute la accion solicitada en cada pregunta.
"""


def pregunta_01():
    """
    La información requerida para este laboratio esta almacenada en el
    archivo "files/input.zip" ubicado en la carpeta raíz.
    Descomprima este archivo.

    Como resultado se creara la carpeta "input" en la raiz del
    repositorio, la cual contiene la siguiente estructura de archivos:


    ```
    train/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    test/
        negative/
            0000.txt
            0001.txt
            ...
        positive/
            0000.txt
            0001.txt
            ...
        neutral/
            0000.txt
            0001.txt
            ...
    ```

    A partir de esta informacion escriba el código que permita generar
    dos archivos llamados "train_dataset.csv" y "test_dataset.csv". Estos
    archivos deben estar ubicados en la carpeta "output" ubicada en la raiz
    del repositorio.

    Estos archivos deben tener la siguiente estructura:

    * phrase: Texto de la frase. hay una frase por cada archivo de texto.
    * sentiment: Sentimiento de la frase. Puede ser "positive", "negative"
      o "neutral". Este corresponde al nombre del directorio donde se
      encuentra ubicado el archivo.

    Cada archivo tendria una estructura similar a la siguiente:

    ```
    |    | phrase                                                                                                                                                                 | target   |
    |---:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------|
    |  0 | Cardona slowed her vehicle , turned around and returned to the intersection , where she called 911                                                                     | neutral  |
    |  1 | Market data and analytics are derived from primary and secondary research                                                                                              | neutral  |
    |  2 | Exel is headquartered in Mantyharju in Finland                                                                                                                         | neutral  |
    |  3 | Both operating profit and net sales for the three-month period increased , respectively from EUR16 .0 m and EUR139m , as compared to the corresponding quarter in 2006 | positive |
    |  4 | Tampere Science Parks is a Finnish company that owns , leases and builds office properties and it specialises in facilities for technology-oriented businesses         | neutral  |
    ```


    """
    import os
    import zipfile
    from pathlib import Path

    try:
        import pandas as pd
    except Exception:  # pragma: no cover - pandas should be available in the test environment
        raise

    repo_root = Path(__file__).resolve().parents[1]

    zip_path = repo_root / "files" / "input.zip"
    # Extract zip into repo root (will create `input/` folder)
    if not zip_path.exists():
        raise FileNotFoundError(f"Expected zip file at {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=repo_root)

    out_dir = repo_root / "files" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(split_name: str):
        rows = []
        base = repo_root / "input" / split_name
        if not base.exists():
            # nothing to do
            return pd.DataFrame(columns=["phrase", "target"])

        # detect categories (directories) and sort for determinism
        categories = [p.name for p in base.iterdir() if p.is_dir()]
        categories = sorted(categories)

        for cat in categories:
            cat_dir = base / cat
            # sort file names for determinism
            files = sorted([p for p in cat_dir.iterdir() if p.is_file()])
            for fpath in files:
                try:
                    text = fpath.read_text(encoding="utf-8").strip()
                except Exception:
                    # fallback with latin-1 if encoding differs
                    text = fpath.read_text(encoding="latin-1").strip()

                # replace newlines with spaces so phrase is a single line
                text = " ".join(text.splitlines())
                rows.append({"phrase": text, "target": cat})

        return pd.DataFrame(rows, columns=["phrase", "target"]) 

    train_df = build_dataset("train")
    test_df = build_dataset("test")

    train_df.to_csv(out_dir / "train_dataset.csv", index=False)
    test_df.to_csv(out_dir / "test_dataset.csv", index=False)

    
