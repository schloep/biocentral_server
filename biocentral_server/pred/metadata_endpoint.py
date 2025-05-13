import dataclasses

from flask import Blueprint, jsonify
from biotrainer.protocols import Protocol

from .models import AvailableModels

prediction_metadata_route = Blueprint('prediction_service', __name__)


@dataclasses.dataclass
class ModelMetadata:
    protocol: Protocol
    description: str
    authors: str
    model_link: str
    citation: str
    licence: str
    description_return_values: str
    model_size: str
    testset_performance: str
    training_data_link: str
    embedder: str

    def to_dict(self):
        return {
            "protocol": self.protocol.name,
            "description": self.description,
            "authors": self.authors,
            "model_link": self.model_link,
            "citation": self.citation,
            "licence": self.licence,
            "description_return_values": self.description_return_values,
            "model_size": self.model_size,
            "testset_performance": self.testset_performance,
            "training_data_link": self.training_data_link,
            "embedder": self.embedder
        }


def get_metadata():
    LightAttention_metadata = ModelMetadata(
        protocol=Protocol.residues_to_class,
        description='',
        authors='Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard',
        model_link='https://github.com/HannesStark/protein-localization',
        citation=' https://doi.org/10.1093/bioadv/vbab035',
        licence='Apache License',
        description_return_values='',
        model_size='',  # onnx in MB
        testset_performance='',
        training_data_link='http://data.bioembeddings.com/public/design/',
        embedder='Rostlab/prot_t5_xl_uniref50'
    )

    TMbed_metadata = ModelMetadata(
        protocol=Protocol.residue_to_class,
        description='',
        authors='Bernhofer, Michael and Rost, Burkhard',
        model_link='https://github.com/BernhoferM/TMbed',
        citation='https://doi.org/10.1101/2022.06.12.495804',
        licence='Apache License',
        description_return_values='',
        model_size='',
        testset_performance='',
        training_data_link='http://data.bioembeddings.com/public/design/',
        embedder='Rostlab/prot_t5_xl_uniref50'
    )

    Conservation_metadata = ModelMetadata(
        protocol=Protocol.residue_to_class,
        description='',
        authors='C{\'{e}}line Marquet and Michael Heinzinger and Tobias Olenyi and Christian Dallago and Kyra Erckert and Michael Bernhofer and Dmitrii Nechaev and Burkhard Rost',
        model_link='https://github.com/Rostlab/VESPA',
        citation='https://doi.org/10.1007/s00439-021-02411-y',
        licence='Apache License',
        description_return_values='',
        model_size='',
        testset_performance='',
        training_data_link='http://data.bioembeddings.com/public/design/',
        embedder='Rostlab/prot_t5_xl_uniref50'
    )
    SecondaryStructure_metadata = ModelMetadata(
        protocol=Protocol.residue_to_class,
        description='',
        authors='',
        model_link='https://github.com/agemagician/ProtTrans',
        citation='https://doi.org/10.1109/TPAMI.2021.3095381',
        licence='Apache License',
        description_return_values='',
        model_size='',
        testset_performance='',
        training_data_link='http://data.bioembeddings.com/public/design/',
        embedder='Rostlab/prot_t5_xl_uniref50'
    )
    BindEmbeDL_metadata = ModelMetadata(
        protocol=Protocol.residue_to_class,
        description='',
        authors='Littmann, Maria and Heinzinger, Michael and Dallago, Christian and Weissenow, Konstantin and Rost, Burkhard',
        model_link='https://github.com/Rostlab/bindPredict/tree/e9f1f33c5b614966fbf7d85b79f856b68ca495ad',
        citation='https://doi.org/10.1038/s41598-021-03431-4',
        licence='Apache License',
        description_return_values='',
        model_size='',
        testset_performance='',
        training_data_link='http://data.bioembeddings.com/public/design/',
        embedder='Rostlab/prot_t5_xl_uniref50'
    )
    SETH_metadata = ModelMetadata(
        protocol=Protocol.residue_to_class,
        description='',
        authors='Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard',
        model_link='https://github.com/DagmarIlz/SETH',
        citation='https://doi.org/10.1101/2022.06.23.497276 ',
        licence='Apache License',
        description_return_values='',
        model_size='',
        testset_performance='',
        training_data_link='http://data.bioembeddings.com/public/design/',
        embedder='Rostlab/prot_t5_xl_uniref50'
    )
    VespaG_metadata = ModelMetadata(
        protocol=Protocol.residue_to_class,  # ?
        description='',
        authors='',
        model_link='https://iteragit.iteratec.de/biocentral-at-iteratec/vespag/-/blob/export_onnx/README.md?ref_type=heads',
        citation='https://doi.org/10.1093/bioinformatics/btae621',
        licence='GNU GENERAL PUBLIC LICENSE',
        description_return_values='',
        model_size='',
        testset_performance='',
        training_data_link='https://zenodo.org/records/11085958',
        embedder='facebook/esm2_t36_3B_UR50D'
    )
    available_models = {
        AvailableModels.LightAttention.name: LightAttention_metadata,
        AvailableModels.TMbed.name: TMbed_metadata,
        AvailableModels.Conservation.name: Conservation_metadata,
        AvailableModels.SecondaryStructure.name: SecondaryStructure_metadata,
        AvailableModels.BindEmbeDL.name: BindEmbeDL_metadata,
        AvailableModels.SETH.name: SETH_metadata,
        AvailableModels.VespaG.name: VespaG_metadata,
    }

    return available_models


def get_metadata_as_dict():
    available_models = get_metadata()
    available_models_dicts = {}
    for model_name, metdata in available_models.items():
        available_models_dicts[model_name] = metdata.to_dict()
    return available_models_dicts


# Endpoint for ProtSpace dimensionality reduction methods for sequences
@prediction_metadata_route.route('/prediction_service/metadata', methods=['GET'])
def metadata():
    return jsonify(get_metadata())
