from matchify.models.base_model import ERBaseModel
from matchify.models.exact_match_model import ExactMatchModel
from matchify.models.flex_match_model import FlexMatchModel
from matchify.models.mlp_match_model import MLPMatchModel
from matchify.utils.data_splitter import DataSplitter
from matchify.pipelines.entity_resolution_pipeline import EntityResolutionPipeline

__version__ = "1.1.0"

__all__ = [
    "ERBaseModel",
    "ExactMatchModel",
    "FlexMatchModel",
    "MLPMatchModel",
    "DataSplitter",
    "EntityResolutionPipeline",
]

# BertMatchModel and SiameseMatchModel are intentionally not imported at
# package-load time: they require the [deep] extra. Import them directly:
#   from matchify.models.bert_match_model import BertMatchModel
#   from matchify.models.siamese_match_model import SiameseMatchModel
