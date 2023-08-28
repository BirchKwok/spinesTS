from io import StringIO

import pandas as pd
from pandas import get_option
from pandas.io.formats import format as fmt


class DataTS(pd.DataFrame):
    """provide common code for spinesTS.data module. """
    def __init__(self, dataset, index=None, columns=None, dtype=None, name=None, date_column=None, format=None):
        super().__init__(data=dataset, index=index, columns=columns, dtype=dtype)

        self.dataset_name = name or 'spinesTS.data.DataTS'
        if date_column is not None:
            self[date_column] = pd.to_datetime(self[date_column], format=format)

    def __str__(self):
        return f"{self.dataset_name} dataset, shape={self.shape}, head_data=\n{self.head()}"

    def _repr__(self):
        return self.__str__()

    def _repr_html_(self) -> str | None:
        """
        Return a html representation for a particular DataFrame.

        Mainly for IPython notebook.
        """
        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return f"<pre>{val}</pre>"

        if get_option("display.notebook_repr_html"):
            min_rows = get_option("display.min_rows")
            max_cols = get_option("display.max_columns")
            show_dimensions = get_option("display.show_dimensions")

            formatter = fmt.DataFrameFormatter(
                self.head(),
                columns=None,
                col_space=None,
                na_rep="NaN",
                formatters=None,
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                escape=True,
                max_rows=5,
                min_rows=min_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=".",
            )

            prefix = f"<div><b>{self.dataset_name}</b>"
            postfix = f"shape: {self.shape}</div>"
            return prefix + fmt.DataFrameRenderer(formatter).to_html(notebook=True) + postfix
        else:
            return None

