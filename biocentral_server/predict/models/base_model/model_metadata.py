from dataclasses import dataclass
from biotrainer.protocols import Protocol


@dataclass
class ModelMetadata:
    name: str
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
            "name": self.name,
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
