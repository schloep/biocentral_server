from flask import request, Blueprint, jsonify

prediction_metadata_route = Blueprint("prediction_service", __name__)

# Endpoint for ProtSpace dimensionality reduction methods for sequences
@prediction_metadata_route.route('/prediction_service/metadata', methods=['GET'])
def metadata():

    available_models = {
        "LightAttention": {
            "protocol": "per-sequence",
            "description": "",
            "authors": "Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard",
            "model_link": "https://github.com/HannesStark/protein-localization",
            "citation":"https://doi.org/10.1093/bioadv/vbab035",
            "licence": "Apache License",
            "description_return_values": "",
            "model_size": "", # onnx in MB
            "testset_performance": "",
            "training_data_link": "http://data.bioembeddings.com/public/design/",
            "embedder": "ProtT5"
        },
        "TMbed": {
            "protocol": "per-residue",
            "description": "",
            "authors": "Bernhofer, Michael and Rost, Burkhard",
            "model-link": "https://github.com/BernhoferM/TMbed",
            "citation":"https://doi.org/10.1101/2022.06.12.495804 ",
            "licence": "Apache License",
            "description_return_values": "",
            "model_size": "",
            "testset_performance": "",
            "training_data_link": "http://data.bioembeddings.com/public/design/",
            "embedder": "ProtT5"
        },
        "Conservation": {
            "protocol": "per-residue",
            "description": "",
            "authors": "C{\'{e}}line Marquet and Michael Heinzinger and Tobias Olenyi and Christian Dallago and Kyra Erckert and Michael Bernhofer and Dmitrii Nechaev and Burkhard Rost",
            "model-link": "https://github.com/Rostlab/VESPA",
            "citation":"https://doi.org/10.1007/s00439-021-02411-y",
            "licence": "Apache License",
            "description_return_values": "",
            "model_size": "",
            "testset_performance": "",
            "training_data_link": "http://data.bioembeddings.com/public/design/",
            "embedder": "ProtT5"
        },
        "Secondary_Structure": {
            "protocol": "per-residue",
            "description": "",
            "authors": "",
            "model-link": "https://github.com/agemagician/ProtTrans",
            "citation":"https://doi.org/10.1109/TPAMI.2021.3095381",
            "licence": "Apache License",
            "description_return_values": "",
            "model_size": "",
            "testset_performance": "",
            "training_data_link": "http://data.bioembeddings.com/public/design/",
            "embedder": "ProtT5"
        },
        "BindEmbeDL": {
            "protocol": "per-residue",
            "description": "",
            "authors": "Littmann, Maria and Heinzinger, Michael and Dallago, Christian and Weissenow, Konstantin and Rost, Burkhard",
            "model-link": "https://github.com/Rostlab/bindPredict/tree/e9f1f33c5b614966fbf7d85b79f856b68ca495ad",
            "citation":"https://doi.org/10.1038/s41598-021-03431-4",
            "licence": "Apache License",
            "description_return_values": "",
            "model_size": "",
            "testset_performance": "",
            "training_data_link": "http://data.bioembeddings.com/public/design/",
            "embedder": "ProtT5"
        },
        "SETH": {
            "protocol": "per-residue",
            "description": "",
            "authors": "Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard",
            "model-link": "https://github.com/DagmarIlz/SETH",
            "citation":"https://doi.org/10.1101/2022.06.23.497276 ",
            "licence": "Apache License",
            "description_return_values": "",
            "model_size": "",
            "testset_performance": "",
            "training_data_link": "http://data.bioembeddings.com/public/design/",
            "embedder": "ProtT5"
        },
        "ESM2VespaG": {
            "protocol": "per-residue", #?
            "description": "",
            "authors": "",
            "model-link": "https://iteragit.iteratec.de/biocentral-at-iteratec/vespag/-/blob/export_onnx/README.md?ref_type=heads",
            "citation":"https://doi.org/10.1093/bioinformatics/btae621",
            "licence": "GNU GENERAL PUBLIC LICENSE",
            "description_return_values": "",
            "model_size": "",
            "testset_performance": "",
            "training_data_link": "https://zenodo.org/records/11085958",
            "embedder": "ESM2"
        },
    }

    return jsonify(available_models)
