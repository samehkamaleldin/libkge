from abc import ABCMeta, abstractmethod


class IExportable(ABCMeta):

    def __init__(self):
        """

        """
        pass

    @abstractmethod
    def export_to_file(self, filepath):
        """ Export model to file.

        Parameters
        ----------
        filepath: str
            file path

        Returns
        -------

        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def import_from_file(self, filepath):
        """ Import model from file

        Parameters
        ----------
        filepath

        Returns
        -------

        """
        raise NotImplementedError("Not implemented")