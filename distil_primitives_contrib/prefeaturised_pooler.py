#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import logging
import os
import math
import typing

import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer

import version

__all__ = ("PrefeaturisedPoolingPrimitive",)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=256,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="inference batch size",
    )
    height = hyperparams.Hyperparameter[typing.Optional[int]](
        default=4,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Height of pooled images",
    )
    width = hyperparams.Hyperparameter[typing.Optional[int]](
        default=4,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Width of pooled images",
    )


class PrefeaturisedPoolingPrimitive(
    transformer.TransformerPrimitiveBase[
        container.DataFrame, container.DataFrame, Hyperparams
    ]
):
    """
    Made specifically to take unpooled outputs from RemoteSensingPretrainedPrimitive
    and pool them.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "825ea1fb-90b2-442c-9905-efba48872102",
            "version": version.__version__,
            "name": "Prefeaturised Pooler",
            "python_path": "d3m.primitives.remote_sensing.remote_sensing_pretrained.PrefeaturisedPooler",
            "source": {
                "name": "Distil",
                "contact": "mailto:vkorapaty@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives-contrib/blob/main/main/distil_primitives_contrib/prefeaturised_pooler.py",
                    "https://github.com/uncharted-distil/distil-primitives-contrib",
                ],
            },
            "installation": [
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives-contrib.git@{git_commit}#egg=distil-primitives-contrib".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.MOMENTUM_CONTRAST,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.REMOTE_SENSING,
        },
    )

    def produce(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> base.CallResult[container.DataFrame]:

        df = inputs.select_columns(
            inputs.metadata.list_columns_with_semantic_types(
                ("http://schema.org/Float",)
            )
        )
        df = df.to_numpy().reshape(
            df.shape[0], 2048, self.hyperparams["height"], self.hyperparams["width"]
        )
        all_img_features = []
        batch_size = self.hyperparams["batch_size"]
        spatial_a = 2.0
        spatial_b = 2.0
        for i in range(math.ceil(df.shape[0] / batch_size)):
            features = df[i * batch_size : (i + 1) * batch_size]
            spatial_weight = features.sum(axis=1, keepdims=True)
            z = (spatial_weight ** spatial_a).sum(axis=(2, 3), keepdims=True)
            z = z ** (1.0 / spatial_a)
            spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

            _, c, w, h = features.shape
            nonzeros = (features != 0).astype(float).sum(axis=(2, 3)) / 1.0 / (
                w * h
            ) + 1e-6
            channel_weight = np.log(nonzeros.sum(axis=1, keepdims=True) / nonzeros)

            features = features * spatial_weight
            features = features.sum(axis=(2, 3))
            features = features * channel_weight
            all_img_features.append(features)
        all_img_features = np.vstack(all_img_features)
        col_names = [f"feat_{i}" for i in range(0, all_img_features.shape[1])]
        feature_df = pd.DataFrame(all_img_features, columns=col_names)

        outputs = container.DataFrame(feature_df.head(1), generate_metadata=True)
        outputs.metadata = outputs.metadata.update(
            (metadata_base.ALL_ELEMENTS,),
            {"dimension": {"length": feature_df.shape[0]}},
        )
        outputs = outputs.append(feature_df.iloc[1:])
        for idx in range(outputs.shape[1]):
            outputs.metadata = outputs.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, idx), "http://schema.org/Float"
            )

        return base.CallResult(outputs)
