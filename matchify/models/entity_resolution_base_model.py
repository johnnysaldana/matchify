class EntityResolutionBaseModel:
    def __init__(self, **kwargs):
        """
        Initialize the EntityResolutionModel object.

        Parameters:
            **kwargs: any additional arguments to be passed to the model.
        """
        self.model = None
        self.kwargs = kwargs

    def fit(self, data):
        """
        Fit the entity resolution model to the input data.

        Parameters:
            data (pandas DataFrame): the input data to fit the model to.

        Returns:
            None
        """
        # Implement fit method in subclass
        pass

    def predict(self, data):
        """
        Predict the entity resolution matches for the input data.

        Parameters:
            data (pandas DataFrame): the input data to predict matches for.

        Returns:
            pandas DataFrame: a DataFrame with columns representing the two records and a match score.
        """
        # Implement predict method in subclass
        pass
