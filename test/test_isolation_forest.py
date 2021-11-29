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

import unittest
from os import path

from distil_primitives_contrib.isolation_forest import IsolationForestPrimitive
import utils as test_utils


class IsolationForestPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "tabular_dataset_2"))

    def test_basic(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")
        dataframe.drop(columns=["delta", "echo"], inplace=True)

        hyperparams_class = IsolationForestPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace({"n_jobs": -1})

        isp = IsolationForestPrimitive(hyperparams=hyperparams)
        isp.set_training_data(
            inputs=dataframe[["alpha", "bravo"]],
        )
        isp.fit()
        results = isp.produce(inputs=dataframe[["alpha", "bravo"]]).value

        self.assertListEqual(
            list(results["outlier_label"]), [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        )


if __name__ == "__main__":
    unittest.main()
