import numpy as np
from bidict import bidict
from collections.abc import Iterable


class KgDataset:

    def __init__(self, triples=None, name="kg"):
        """ Create new instance of class KgDataset

        Parameters
        -----------

        triples: ndarray
            array of triples with size (?, 3)
        name: str
            name of the dataset
        """
        self.name = name
        self.ent_mappings = bidict()
        self.rel_mappings = bidict()
        self.data = dict()
        self.metadata = dict()
        self.metadata['name'] = name

        if triples is not None:
            self.load_triples(triples, tag="default-all")

    def load_triples(self, triplets, tag="default"):
        """ append triplets' entities and relations into the knowledge graph dictionary

        Parameters
        ----------
        triplets : list
            array of triplets with size (?, 3)
        tag : str
            triples data tag

        Returns
        -------
        KgDataset
            dataset object
        """
        for s, p, o in triplets:
            if s not in self.ent_mappings:
                self.ent_mappings[s] = len(self.ent_mappings.keys())
            if p not in self.rel_mappings:
                self.rel_mappings[p] = len(self.rel_mappings.keys())
            if o not in self.ent_mappings:
                self.ent_mappings[o] = len(self.ent_mappings.keys())

        if tag in self.data:
            tag = tag + "1"
        self.data[tag] = self.labels2indices(triplets)

        return self.data[tag]

    def labels2indices(self, triplets):
        """
        transform triplets from label form to indices form

        :param triplets: ndarray
            array of textual triplets with size (?, 3)
        :return: ndarray
            array of index-based triplets with size (?, 3)
        """
        out = []
        for s, p, o in triplets:
            out.append([self.ent_mappings[s], self.rel_mappings[p], self.ent_mappings[o]])
        return np.array(out)

    def indices2labels(self, triplets):
        """
        transform triplets from indices form to label form

        :param triplets: ndarray
            array of index-based triplets with size (?, 3)
        :return: ndarray
            array of textual triplets with size (?, 3)
        """
        out = []
        for s, p, o in triplets:
            out.append([self.ent_mappings.inv[s], self.rel_mappings.inv[p], self.ent_mappings.inv[o]])
        return np.array(out)

    def get_ent_indices(self, ent_labels):
        """ Get entity index/indices of a given entity label/labels

        Parameters
        ----------
        ent_labels : np.ndarray
            entity labels array

        Returns
        -------
        np.array
            entity indices array
        """
        if type(ent_labels) == str:
            return self.ent_mappings[ent_labels]
        else:
            return np.array([self.ent_mappings[l] for l in ent_labels])

    def get_ent_labels(self, ent_indices):
        """ Get entity label/labels of a given entity index/indices

        Parameters
        ----------
        ent_indices : np.ndarray
            entity indices array

        Returns
        -------
        np.array
            entity labels array
        """
        if type(ent_indices) == int:
            return self.ent_mappings.inv[ent_indices]
        else:
            return np.array([self.ent_mappings.inv[l] for l in ent_indices])

    def get_rel_indices(self, rel_labels):
        """ Get relation index/indices of a given relation label/labels

        Parameters
        ----------
        rel_labels : Iterable
            relation labels iterable

        Returns
        -------
        np.array
            relation indices array
        """
        if isinstance(rel_labels, Iterable):
            return np.array([self.rel_mappings[l] for l in rel_labels])
        else:
            return self.rel_mappings[str(rel_labels)]

    def get_rel_labels(self, rel_indices):
        """ Get relation label/labels of a given relation index/indices

        Parameters
        ----------
        rel_indices : Iterable
            relation indices iterable

        Returns
        -------
        np.array
            relation labels array
        """
        if isinstance(rel_indices, Iterable):
            return np.array([self.rel_mappings.inv[l] for l in rel_indices])
        else:
            return self.rel_mappings.inv[int(rel_indices)]

    def get_ents_count(self):
        """ Get the number of entities in the dataset

        Returns
        -------
        int
            The number of entities in the dataset
        """
        return len(self.ent_mappings.keys())

    def get_rels_count(self):
        """ Get the number of relations in the dataset

        Returns
        -------
        int
            The number of relations in the dataset
        """
        return len(self.rel_mappings.keys())
