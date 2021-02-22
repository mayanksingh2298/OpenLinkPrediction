"""
A :class:`~allennlp_data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from allennlp_data.fields.field import Field
from allennlp_data.fields.adjacency_field import AdjacencyField
from allennlp_data.fields.tensor_field import TensorField
from allennlp_data.fields.flag_field import FlagField
from allennlp_data.fields.index_field import IndexField
from allennlp_data.fields.label_field import LabelField
from allennlp_data.fields.list_field import ListField
from allennlp_data.fields.metadata_field import MetadataField
from allennlp_data.fields.multilabel_field import MultiLabelField
from allennlp_data.fields.namespace_swapping_field import NamespaceSwappingField
from allennlp_data.fields.sequence_field import SequenceField
from allennlp_data.fields.sequence_label_field import SequenceLabelField
from allennlp_data.fields.span_field import SpanField
from allennlp_data.fields.text_field import TextField
from allennlp_data.fields.array_field import ArrayField
