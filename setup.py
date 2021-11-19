from distutils.core import setup

from setuptools import find_packages
from version import __version__

with open("version.py") as f:
    exec(f.read())

setup(
    name="distil-primitives-contrib",
    version=__version__,
    description="Additional Distil primitives as a single library",
    packages=find_packages(),
    keywords=["d3m_primitive"],
    license="Apache-2.0",
    install_requires=[
        "d3m",  # d3m best-practice moving forward is to remove the version (simplifies updates)
        # shared d3m versions - need to be aligned with core package
        "scikit-learn==0.22.2.post1",
        "numpy==1.18.2",
        "pandas>=1.1.3",
        # additional dependencies
        "joblib>=0.13.2",
        "haversine==2.3.1",
        "rapidfuzz==1.5.1"
    ],
    entry_points={
        "d3m.primitives": [
            "classification.isolation_forest.IsolationForestPrimitive = distil_primitives_contrib.isolation_forest:IsolationForestPrimitive",
            "data_transformation.vector_bounds_filter.DistilVectorBoundsFilter = distil_primitives_contrib.vector_filter:VectorBoundsFilterPrimitive",
            "data_transformation.concat.DistilVerticalConcat = distil_primitives_contrib.concat:VerticalConcatenationPrimitive",
            "data_transformation.time_series_binner.DistilTimeSeriesBinner = distil_primitives_contrib.time_series_binner:TimeSeriesBinnerPrimitive",
            "remote_sensing.remote_sensing_pretrained.PrefeaturisedPooler = distil_primitives_contrib.prefeaturised_pooler:PrefeaturisedPoolingPrimitive",
            "data_transformation.fuzzy_join.DistilFuzzyJoin = distil_primitives_contrib.fuzzy_join:FuzzyJoinPrimitive",
            "feature_selection.mutual_info_classif.DistilMIRanking = distil_primitives_contrib.mi_ranking:MIRankingPrimitive",
        ],
    },
)
