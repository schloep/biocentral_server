import torch
import numpy as np
from dataclasses import dataclass

from .base_model import BaseModel
from ..utils import load_onnx_model
from .additional.score_normalizer import ScoreNormalizer
from .additional.mutations import compute_mutation_score, mask_non_mutations, SAV


class VespaG(BaseModel):
    name = 'VespaG'

    def __init__(self, batch_size):
        super().__init__(batch_size=batch_size)
        self.model = load_onnx_model(model_name=self.name)
        self.GEMME_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
        self.AMINO_ACIDS = sorted(self.GEMME_ALPHABET)
        zero_based_mutations = False
        # TODO: how to obtain the sequences in this desired format? Or use "read_mutation_file() instead?"
        self.mutations_per_protein = {
            protein_id: [
                SAV(i, wildtype_aa, other_aa, not zero_based_mutations)
                for i, wildtype_aa in enumerate(sequence)
                for other_aa in self.AMINO_ACIDS
                if other_aa != wildtype_aa
            ]
            for protein_id, sequence in sequences.items()
        }
        self.normalizer = ScoreNormalizer('minmax')

    def _prepare_inputs(self, embeddings):
        return [{'input': embedding.unsqueeze(0).numpy()} for embedding in embeddings.values()]

    def predict(self, embeddings):
        inputs = self._prepare_inputs(embeddings=embeddings)
        embedding_ids = embeddings.keys()
        vespag_scores = {}
        for i, sequence_embedding in enumerate(inputs):
            y = self.model.run(None, sequence_embedding)
            y = torch.from_numpy(np.float32(np.stack(y[0]))).squeeze(0)
            y = mask_non_mutations(y, embeddings[embedding_ids[i]])
            vespag_scores[id] = y.detach().numpy()
        self.normalizer.fit(np.concatenate([y.flatten() for y in vespag_scores.values()]))
        return self._post_process(model_output=vespag_scores, embedding_ids=embedding_ids)

    def _post_process(self, model_output, embedding_ids):
        scores_per_protein = {}
        transform_scores = True  # default value in the VespaG repo
        for id, y in model_output.items():
            scores_per_protein[id] = {
                mutation: compute_mutation_score(
                    y,
                    mutation,
                    transform=transform_scores,
                    normalizer=self.normalizer,
                )
                for mutation in self.mutations_per_protein[id]
            }
