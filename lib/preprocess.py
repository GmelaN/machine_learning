import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin, BaseEstimator

import seaborn as sns
import matplotlib.pyplot as plt

from math import ceil

from warnings import warn
from typing import Callable, Literal, List, Union
from typing_extensions import Protocol


__version__ = "0.7"


class ScikitModel(Protocol):
    def fit(self, X, y): ...
    def predict(self, X): ...
    def score(self, X, y): ...
    def set_params(self, **params): ...


class Preprocess():
    def __init__(self, data: pd.DataFrame, categorial_threshold: int=30) -> None:
        self.data = data
        self.pipeline_args = None

        self.categorial_threshold = categorial_threshold
        self._update_numeric_data()

    def train_test_split(self, y_label, stratify:bool=True, test_size:float=0.2, random_state:(int|None)=36) -> List:
        '''
        same as train_test_split.
        '''

        if y_label not in self.data.keys():
            raise ValueError("{y_label} is not found on data.")

        X_train, X_test, y_train, y_test = train_test_split(
            self.data.drop(y_label, axis=1),
            self.data[y_label],
            test_size=test_size,
            random_state=random_state,
            stratify=self.data[y_label] if stratify else None,
            )

        return [X_train, X_test, y_train, y_test]
    

    def submit(self, model: ScikitModel, preprocess_args: dict, path: str, export_file_name: str, example_file_name: str, target: str):
        # 전처리 수행
        for key in preprocess_args.keys():
            if key in dir(self):
                getattr(self, key)(**preprocess_args[key])

        predictions = model.predict(self.data)
        submission = pd.read_csv(path + example_file_name, index_col=[0])

        submission[target] = predictions
        submission[target] = submission[target].round(2)
        submission.to_csv(path + export_file_name)


    def encode_data(self, method: Literal["OneHotEncoder"], exclude: (List[str] | None)) -> None:
        '''
        encodes categorial data.

        arguments
        ---------
        `method`: str, "OneHotEncoder" or "LabelEncoder".
        `exclude`: List[str] | None, column name of data to exclude encoding.

        returns
        -------
        None, inner dataset will be modified.
        '''

        '''
        TO-DO: LabelEncoder 지원
        '''

        # 유효성 검증
        SUPPORTED_METHODS = ("OneHotEncoder", "LabelEncoder")

        if method not in SUPPORTED_METHODS:
            raise ValueError("{method} is not supported encoding method.")
        
        if exclude is not None:
            for e in exclude:
                if e not in self.data.keys():
                    raise ValueError("{target_data} is not found in dataset.")


        # 데이터 상태 업데이트
        self._update_numeric_data()

        encoder = OneHotEncoder(sparse_output=False)

        # 카테고리형 데이터 - 속성의 가짓수가 {self.categorial_threshold}개 미만이면 카테고리형으로 판단
        categorial_data_keys = [i for i in self._object_data.keys()]
        categorial_data_keys += list(self._categorial_data)

        # argument로 선택된 제외 대상 데이터 drop
        if exclude is not None:
            for e in exclude:
                if e in categorial_data_keys:
                    categorial_data_keys.remove(e)

        # 인코딩을 적용하지 않는 다른 데이터들
        no_encode_data = self.data.drop(categorial_data_keys, axis=1)

        # 인코딩 된 데이터가 들어갈 데이터프레임
        encoded_data = pd.DataFrame()

        # 각 데이터 인코드
        for key in categorial_data_keys:
            encoded_raw_data = encoder.fit_transform(self.data[[key]])
            encoded_raw_data = encoded_raw_data.astype(np.int8)

            # 인코딩 된 데이터 라벨 붙여주기
            encoded_data_columns = [f"{key}_{i}" for i in encoder.categories_[0]]
            encoded_data_concat = pd.DataFrame(encoded_raw_data, index=self.data[key].index, columns=encoded_data_columns)

            encoded_data = pd.concat([
                encoded_data,
                encoded_data_concat,
            ], axis=1)

        # 데이터 타입 지정
        encoded_data = encoded_data.astype("category")

        # 다른 데이터와 합치기
        self.data = pd.concat([no_encode_data, encoded_data], axis=1)

        self._update_numeric_data()


    def scale(self, method: Literal["StandardScaler", "MinMaxScaler", "RobustScaler"]="MinMaxScaler", exclude: List[str]|None=None) -> None:
        '''
        scale numeric continuous data, will not process categorial data.

        args
        ----
        `method`=Literal["StandardScaler", "MinMaxScaler", "RobustScaler"] = "MinMaxScaler", scaling method.
        `exclude`=list[str] | None = None, excluded attribute's name from scaling.

        returns
        -------
        None, inner dataset will be modified.
        '''

        # 유효성 검증
        SUPPORTED_METHODS = ("StandardScaler", "MinMaxScaler", "RobustScaler")

        if method not in SUPPORTED_METHODS:
            raise ValueError("{method} is not supported scaling method.")

        if exclude is not None:
            for e in exclude:
                if e not in self.data.keys():
                    raise ValueError("{e} is not found on data.")

        # 데이터 상태 업데이트
        self._update_numeric_data()


        # 연속형 변수에 대하여 스케일 진행
        scaler = globals()[method]()

        target = self._numeric_data

        if exclude is not None:
            for e in exclude:
                if e in target.keys():
                    target = target.drop(exclude, axis=1)

        scaled = scaler.fit_transform(target)
        numeric_data_scaled = pd.DataFrame(scaled, index=target.index, columns=target.columns)

        for key in numeric_data_scaled.keys():
            self.data.loc[:, key] = numeric_data_scaled.loc[:, key]

        self._update_numeric_data()


    def get_polynormial_features(self, degree:int=2, exclude: List[str]|None=None) -> None:
        '''
        scale numeric continuous data, will not process categorial data.

        args
        ----
        `degree`: int=2, degree of polynormial features.
        `exclude`=list[str] | None = None, excluded attribute's name from the target of polynormial featuring.

        returns
        -------
        None, inner dataset will be modified.
        '''

        # 유효성 검증
        if degree < 2:
            raise ValueError("degree must be 2 or higher, got {degree}.")
        
        if exclude is not None:
            for e in exclude:
                if e not in self.data.keys():
                    raise ValueError(f"{e} is not found on data.")

        if not self._object_data.empty:
            warn("non-numeric data found, these datas will be ignored.")

        # 데이터 상태 업데이트
        self._update_numeric_data()


        # 연속형 변수에 대하여 다항 변수화 적용
        poly = PolynomialFeatures(degree=degree, include_bias=False)

        if exclude is not None:
            target = self._numeric_data.drop(exclude, axis=1)
        else:
            target = self._numeric_data

        poly_data = poly.fit_transform(target)

        poly_result = pd.DataFrame(poly_data, index=target.index, columns=poly.get_feature_names_out())


        # 제외된 데이터들 파악
        other_data_keys = []

        if exclude is not None:
            other_data_keys += exclude

        other_data_keys += (list(self._categorial_data.keys()) + list(self._object_data.keys()))

        # 데이터 붙이기
        self.data = pd.concat([self.data[other_data_keys], poly_result], axis=1)

        self._update_numeric_data()


    def remove_null(self, method: Literal["ffill", "backfill", "bfill", "drop"], ) -> None:
        '''
        drop or fill null values.
        
        arguments
        ---------
        `method`: ("ffill", "backfill", "bfill") will fill null values on their method, "drop" will drop entire row containing and null values.

        returns
        -------
        None, inner data will be modified.
        '''
        if method == "drop":
            self.data.dropna(how="any", axis=0, inplace=True)

        else:
            self.data.fillna(method=method, inplace=True)


    def remove_outliers(self, method:Literal["IQR"]="IQR", threshold: float = 1.5) -> None:
        '''
        removes outliers by differing from values over/under IQR * threshold.

        arguments
        ---------
        :threshold: threshold for evaluate value overs/unders IQR * threshold + Qn.

        returns
        -------
        None, inner data will be modified.
        '''

        # 파라미터 유효성 검증
        if method != "IQR":
            raise NotImplementedError("currently only IQR method is supported.")

        # 데이터 상태 업데이트
        self._update_numeric_data()


        # 연속형 데이터 가져오기
        continuous_data_keys = list(self._numeric_data.keys())

        # 연속형 데이터에 대하여 사분위수 구하기
        data_continuous = self.data[continuous_data_keys]

        for key in data_continuous.keys():
            Q1 = data_continuous[key].quantile(0.25)
            Q3 = data_continuous[key].quantile(0.75)
            IQR = Q3 - Q1

            # 이 limit 값들을 넘어서는 값들은 이상치로 판단
            upper_limit = Q3 + IQR * threshold
            lower_limit = Q1 - IQR * threshold

            # 이상치 제거 수행
            idx = data_continuous[key][(data_continuous[key] < lower_limit) | (data_continuous[key] > upper_limit)].index
            data_continuous = data_continuous.drop(idx, axis=0)
        
        # 기타 데이터 불러오기
        other_data = self.data.drop(continuous_data_keys, axis=1)

        # 기타 데이터와 이상치를 제거한 데이터 병합
        self.data = pd.concat([
                data_continuous,
                other_data.iloc[data_continuous.index, :],
            ], axis=1)

        # 데이터 정보 업데이트
        self._update_numeric_data()


    def plot_corr_heatmap(self, figure_size: int=12, strategy: Literal["pearson", "kendall", "spearman"]="pearson") -> None:
        '''
        plot correlation matrix heatmap.

        args
        ----
        :figure_size: figure size of each plots.
        :strategy: strategy of generating correation matrix.

        returns
        -------
        None, printing output of plot.
        '''

        plt.figure(figsize=(figure_size, figure_size))
        sns.heatmap(data=self.data.select_dtypes(include="number").corr(), annot=True, fmt=".2f", linewidths=5, cmap="RdBu", vmin=-1, vmax=1)
        plt.show()


    def plot_distribution(self, figure_size: int=12) -> None:
        '''
        plot distribution of each values.

        args
        ----
        `figure_size`: figure size of each plots.

        returns
        -------
        None, printing output of plot.
        '''

        self._update_numeric_data()

        plot_data = self.data.select_dtypes(include="number")


        plt.figure(figsize=(figure_size, figure_size))

        nrows = ceil(len(plot_data.keys()) ** 0.5)
        ncols = ceil(len(plot_data.keys()) / nrows)

        for i in range(1, len(plot_data.keys()) + 1):
            plt.subplot(nrows, ncols, i)
            sns.histplot(plot_data.iloc[:, i-1], kde=True, stat="density")

        plt.tight_layout()
        plt.show()


    def _update_numeric_data(self) -> None:
        '''
        update data categories.
        args
        ----
        None.

        returns
        -------
        None, update inner datasets.
        '''

        # 연속형 데이터
        self._numeric_data = self.data.select_dtypes(include="number")

        # 카테고리형 데이터
        self._categorial_data = pd.DataFrame()

        for key in self._numeric_data.keys():
            if len(self._numeric_data[key].value_counts()) <= self.categorial_threshold:
                self._categorial_data = pd.concat([self._categorial_data, self._numeric_data[key]], axis=1)
                self._numeric_data.drop(key, axis=1, inplace=True)

        # 문자 데이터
        self._object_data = self.data.drop(self._numeric_data.keys(), axis=1)
        self._object_data = self._object_data.drop(self._categorial_data.keys(), axis=1)


    def drop(self, **args):
        self.data = self.data.drop(**args)


    def set_pipeline(self, pipeline_args:dict) -> dict:
        if type(pipeline_args) is not dict:
            raise ValueError(f"{pipeline_args} is not determined as dict.")

        for key in pipeline_args.keys():
            if key not in dir(self):
                raise ValueError(f"{key} is not found in Preprocess attributes.")


        self.preprocess_args = pipeline_args
        return self.preprocess_args


    def execute_pipeliine(self) -> pd.DataFrame:
        if self.preprocess_args is None:
            raise ValueError(f"preprocess pipeline argument is empty. didn't you called set_pipeline?")
        
        # 전처리 수행
        for key in self.preprocess_args.keys():
            if key in dir(self):
                getattr(self, key)(**self.preprocess_args[key])
        
        return self.data
