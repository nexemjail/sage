from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def get_column_transformer() -> ColumnTransformer:
    num_columns = ["Age", "Fare"]
    median_imputer = SimpleImputer(strategy="median")
    cat_columns = ["Sex", "Embarked", "Pclass"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    ct = ColumnTransformer(
        transformers=[
            (
                "num_columns",
                median_imputer,
                num_columns,
            ),
            ("cat_columns", ohe, cat_columns),
        ]
    )
    return ct
