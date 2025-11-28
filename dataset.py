import numpy as np
import pandas as pd


class Dataset:
    def __init__(
        self,
        choices,
        items_features_by_choice,
        items_features_by_choice_names,
        shared_features_by_choice=None,
        shared_features_by_choice_names=None,
        available_items_by_choice=None,
    ):
        # If shared_features_by_choice is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names
        if shared_features_by_choice is not None:
            if not isinstance(shared_features_by_choice, tuple):
                self._return_shared_features_by_choice_tuple = False
                if shared_features_by_choice_names is not None:
                    if len(shared_features_by_choice[0]) != len(
                        shared_features_by_choice_names
                    ):
                        raise ValueError(
                            f"""Number of features given does not match
                                         number of features names given:
                                           {len(shared_features_by_choice[0])} and
                                            {len(shared_features_by_choice_names)}"""
                        )
                else:
                    print(
                        """Shared Features Names were not provided, will not be able to
                                    fit models needing them such as Conditional Logit."""
                    )

                shared_features_by_choice_names = (shared_features_by_choice_names,)
                shared_features_by_choice = (shared_features_by_choice,)

            else:
                self._return_shared_features_by_choice_tuple = True
                if shared_features_by_choice_names is not None:
                    for sub_k, (sub_features, sub_names) in enumerate(
                        zip(shared_features_by_choice, shared_features_by_choice_names)
                    ):
                        if len(sub_features[0]) != len(sub_names):
                            raise ValueError(
                                f"""{sub_k}-th given shared_features_by_choice and
                                shared_features_by_choice_names shapes do not match:
                                {len(sub_features[0])} and {len(sub_names)}."""
                            )
                else:
                    print(
                        """Shared Features Names were not provided, will not be able to
                                    fit models needing them such as Conditional Logit."""
                    )
                    shared_features_by_choice_names = (None,) * len(
                        shared_features_by_choice
                    )
        else:
            self._return_shared_features_by_choice_tuple = False

        # If items_features_by_choice is not given as tuple, transform it internally as a tuple
        # A bit longer because can be None and need to also handle names
        if (
            not isinstance(items_features_by_choice, tuple)
            and items_features_by_choice is not None
        ):
            self._return_items_features_by_choice_tuple = False
            if items_features_by_choice_names is not None:
                if len(items_features_by_choice[0][0]) != len(
                    items_features_by_choice_names
                ):
                    raise ValueError(
                        f"""Number of items_features_by_choice given does not match
                                     number of items_features_by_choice_names given:
                                     {len(items_features_by_choice[0][0])} and
                                     {len(items_features_by_choice_names)}"""
                    )
            else:
                print(
                    """Items Features Names were not provided, will not be able to
                                fit models needing them such as Conditional Logit."""
                )
            items_features_by_choice = (items_features_by_choice,)
            items_features_by_choice_names = (items_features_by_choice_names,)

        # items_features_by_choice is already a tuple, names are given, checking consistency
        elif (
            items_features_by_choice is not None
            and items_features_by_choice_names is not None
        ):
            for sub_k, (sub_features, sub_names) in enumerate(
                zip(items_features_by_choice, items_features_by_choice_names)
            ):
                if len(sub_features[0][0]) != len(sub_names):
                    raise ValueError(
                        f"""{sub_k}-th given items_features_by_choice with names
                        {sub_names} and
                        items_features_by_choice_names shapes do not match:
                        {len(sub_features[0][0])} and {len(sub_names)}."""
                    )
            self._return_items_features_by_choice_tuple = True
        else:
            self._return_items_features_by_choice_tuple = False

        if shared_features_by_choice is not None:
            for i, feature in enumerate(shared_features_by_choice):
                if isinstance(feature, list):
                    shared_features_by_choice = (
                        shared_features_by_choice[:i]
                        + (np.array(feature),)
                        + shared_features_by_choice[i + 1 :]
                    )

        for i, feature in enumerate(items_features_by_choice):
            if isinstance(feature, list):
                items_features_by_choice = (
                    items_features_by_choice[:i]
                    + (np.array(feature),)
                    + items_features_by_choice[i + 1 :]
                )

        if available_items_by_choice is not None:
            if isinstance(available_items_by_choice, list):
                available_items_by_choice = np.array(
                    available_items_by_choice,
                    dtype=object,  # Are you sure ?
                )

        if isinstance(choices, pd.DataFrame):
            if "choice_id" in choices.columns:
                choices = choices.set_index("choice_id")
            choices = choices.loc[np.sort(choices.index)]
            items = np.sort(np.unique(choices.to_numpy()))
            choices = [np.where(items == c)[0] for c in np.squeeze(choices.to_numpy())]
            choices = np.squeeze(choices)
        elif isinstance(choices, pd.Series):
            choices = choices.to_numpy()
        elif isinstance(choices, list):
            choices = np.array(choices)

        self.shared_features_by_choice = shared_features_by_choice
        self.items_features_by_choice = items_features_by_choice
        self.available_items_by_choice = available_items_by_choice
        self.choices = choices

        self.base_num_items = len(np.unique(choices))

        self.shared_features_by_choice_names = shared_features_by_choice_names
        self.items_features_by_choice_names = items_features_by_choice_names

        self._return_types = self._check_types()

        self.indexer = Indexer(self)

    def _check_types(self):
        return_types = []

        shared_features_types = []
        if self.shared_features_by_choice is not None:
            for feature in self.shared_features_by_choice:
                if np.issubdtype(feature[0].dtype, np.integer):
                    shared_features_types.append(np.int32)
                else:
                    shared_features_types.append(np.float32)
        return_types.append(tuple(shared_features_types))

        items_features_types = []
        for items_feat in self.items_features_by_choice:
            if np.issubdtype(items_feat[0].dtype, np.integer):
                items_features_types.append(np.int32)
            else:
                items_features_types.append(np.float32)
        return_types.append(tuple(items_features_types))

        return_types.append(np.float32)
        return_types.append(np.int32)

        return return_types

    def __len__(self):
        return len(self.choices)

    def summary(self):
        print("%=====================================================================%")
        print("%%% Summary of the dataset:")
        print("%=====================================================================%")
        print("Number of items:", self.base_num_items)
        print(
            "Number of choices:",
            len(self),
        )
        print("%=====================================================================%")

        if self.shared_features_by_choice is not None:
            print(" Shared Features by Choice:")
            print(
                f" {sum([f.shape[1] for f in self.shared_features_by_choice])} shared features"
            )
            if self.shared_features_by_choice_names is not None:
                if self.shared_features_by_choice_names[0] is not None:
                    print(f" with names: {self.shared_features_by_choice_names}")
        else:
            print(" No Shared Features by Choice registered")
        print("\n")

        if self.items_features_by_choice is not None:
            if self.items_features_by_choice[0] is not None:
                print(" Items Features by Choice:")
                print(
                    f"""{
                        sum(
                            [
                                f.shape[2] if f.ndim == 3 else 1
                                for f in self.items_features_by_choice
                            ]
                        )
                    } items features """
                )
                if self.items_features_by_choice_names is not None:
                    if self.items_features_by_choice_names[0] is not None:
                        print(f" with names: {self.items_features_by_choice_names}")
        else:
            print(" No Items Features by Choice registered")
        print("%=====================================================================%")

    def get_choices_batch(self, choices_indexes, features=None):
        _ = features
        if isinstance(choices_indexes, list):
            if np.array(choices_indexes).ndim > 1:
                raise ValueError(
                    """ChoiceDataset unidimensional can only be batched along choices
                                 dimension received a list with several axis of indexing."""
                )
            if self.shared_features_by_choice is None:
                shared_features_by_choice = None
            else:
                shared_features_by_choice = list(
                    shared_features_by_choice[choices_indexes]
                    for shared_features_by_choice in self.shared_features_by_choice
                )

            if self.items_features_by_choice is None:
                items_features_by_choice = None
            else:
                items_features_by_choice = list(
                    items_features_by_choice[choices_indexes]
                    for items_features_by_choice in self.items_features_by_choice
                )
            if self.available_items_by_choice is None:
                available_items_by_choice = np.ones(
                    (len(choices_indexes), self.base_num_items)
                ).astype("float32")
            else:
                if isinstance(self.available_items_by_choice, tuple):
                    available_items_by_choice = self.available_items_by_choice[0].batch[
                        self.available_items_by_choice[1][choices_indexes]
                    ]
                else:
                    available_items_by_choice = self.available_items_by_choice[
                        choices_indexes
                    ]

            choices = self.choices[choices_indexes].astype(self._return_types[3])

            if shared_features_by_choice is not None:
                for i in range(len(shared_features_by_choice)):
                    shared_features_by_choice[i] = shared_features_by_choice[i].astype(
                        self._return_types[0][i]
                    )
                if not self._return_shared_features_by_choice_tuple:
                    shared_features_by_choice = shared_features_by_choice[0]
                else:
                    shared_features_by_choice = tuple(shared_features_by_choice)

            if items_features_by_choice is not None:
                for i in range(len(items_features_by_choice)):
                    items_features_by_choice[i] = items_features_by_choice[i].astype(
                        self._return_types[1][i]
                    )
                # items_features_by_choice were not given as a tuple, so we return do not return
                # it as a tuple
                if not self._return_items_features_by_choice_tuple:
                    items_features_by_choice = items_features_by_choice[0]
                else:
                    items_features_by_choice = tuple(items_features_by_choice)

            return (
                shared_features_by_choice,
                items_features_by_choice,
                available_items_by_choice,
                choices,
            )

        if isinstance(choices_indexes, slice):
            return self.get_choices_batch(
                list(range(*choices_indexes.indices(self.choices.shape[0])))
            )

        choices_indexes = [choices_indexes]
        (
            shared_features_by_choices,
            items_features_by_choice,
            available_items_by_choice,
            choice,
        ) = self.get_choices_batch(choices_indexes)
        if shared_features_by_choices is not None:
            if isinstance(shared_features_by_choices, tuple):
                shared_features_by_choices = tuple(
                    feat[0] for feat in shared_features_by_choices
                )
            else:
                shared_features_by_choices = shared_features_by_choices[0]
        if items_features_by_choice is not None:
            if isinstance(items_features_by_choice, tuple):
                items_features_by_choice = tuple(
                    feat[0] for feat in items_features_by_choice
                )
            else:
                items_features_by_choice = items_features_by_choice[0]

        return (
            shared_features_by_choices,
            items_features_by_choice,
            available_items_by_choice[0],
            choice[0],
        )

    def __getitem__(self, choices_indexes):
        if isinstance(choices_indexes, int):
            choices_indexes = [choices_indexes]
        elif isinstance(choices_indexes, slice):
            return self.__getitem__(
                list(range(*choices_indexes.indices(len(self.choices))))
            )

        try:
            if self.shared_features_by_choice[0] is None:
                shared_features_by_choice = None
            else:
                shared_features_by_choice = tuple(
                    self.shared_features_by_choice[i][choices_indexes]
                    for i in range(len(self.shared_features_by_choice))
                )
                if not self._return_shared_features_by_choice_tuple:
                    shared_features_by_choice = shared_features_by_choice[0]
        except TypeError:
            shared_features_by_choice = None

        try:
            if self.items_features_by_choice[0] is None:
                items_features_by_choice = None
            else:
                items_features_by_choice = tuple(
                    self.items_features_by_choice[i][choices_indexes]
                    for i in range(len(self.items_features_by_choice))
                )
                if not self._return_items_features_by_choice_tuple:
                    items_features_by_choice = items_features_by_choice[0]
        except TypeError:
            items_features_by_choice = None

        try:
            if self.shared_features_by_choice_names[0] is None:
                shared_features_by_choice_names = None
            else:
                shared_features_by_choice_names = self.shared_features_by_choice_names
                if not self._return_shared_features_by_choice_tuple:
                    shared_features_by_choice_names = shared_features_by_choice_names[0]
        except TypeError:
            shared_features_by_choice_names = None
        try:
            if self.items_features_by_choice_names[0] is None:
                items_features_by_choice_names = None
            else:
                items_features_by_choice_names = self.items_features_by_choice_names
                if not self._return_items_features_by_choice_tuple:
                    items_features_by_choice_names = items_features_by_choice_names[0]
        except TypeError:
            items_features_by_choice_names = None

        try:
            if isinstance(self.available_items_by_choice, tuple):
                available_items_by_choice = self.available_items_by_choice[1][
                    choices_indexes
                ]
            else:
                available_items_by_choice = self.available_items_by_choice[
                    choices_indexes
                ]
        except TypeError:
            available_items_by_choice = None

        return Dataset(
            shared_features_by_choice=shared_features_by_choice,
            items_features_by_choice=items_features_by_choice,
            available_items_by_choice=available_items_by_choice,
            choices=[self.choices[i] for i in choices_indexes],
            shared_features_by_choice_names=shared_features_by_choice_names,
            items_features_by_choice_names=items_features_by_choice_names,
        )

    @property
    def batch(self):
        return self.indexer

    def iter_batch(self, batch_size, shuffle=False, sample_weight=None):
        if sample_weight is not None and isinstance(sample_weight, list):
            sample_weight = np.array(sample_weight)
        if batch_size == -1 or batch_size == len(self):
            yield self.indexer.get_full_dataset(sample_weight=sample_weight)
        else:
            # Get indexes for each choice
            num_choices = len(self)
            indexes = np.arange(num_choices)

            # Shuffle indexes
            if shuffle and not batch_size == -1:
                indexes = np.random.permutation(indexes)

            yielded_size = 0
            while yielded_size < num_choices:
                # Return sample_weight if not None, for index matching
                batch_indexes = indexes[
                    yielded_size : yielded_size + batch_size
                ].tolist()
                if sample_weight is not None:
                    yield (
                        self.batch[batch_indexes],
                        sample_weight[batch_indexes],
                    )
                else:
                    yield self.batch[batch_indexes]
                yielded_size += batch_size


class Indexer:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def _get_shared_features_by_choice(self, choices_indexes):
        if self.dataset.shared_features_by_choice is None:
            shared_features_by_choice = None
        else:
            shared_features_by_choice = []
            for i, shared_feature in enumerate(self.dataset.shared_features_by_choice):
                if hasattr(shared_feature, "batch"):
                    shared_features_by_choice.append(
                        shared_feature.batch[choices_indexes]
                    )
                else:
                    shared_features_by_choice.append(shared_feature[choices_indexes])
        return shared_features_by_choice

    def _get_items_features_by_choice(self, choices_indexes):
        if self.dataset.items_features_by_choice is None:
            return None
        items_features_by_choice = []
        for i, items_feature in enumerate(self.dataset.items_features_by_choice):
            if hasattr(items_feature, "batch"):
                items_features_by_choice.append(items_feature.batch[choices_indexes])
            else:
                items_features_by_choice.append(items_feature[choices_indexes])
        return items_features_by_choice

    def __getitem__(self, choices_indexes):
        if isinstance(choices_indexes, list):
            shared_features_by_choice = self._get_shared_features_by_choice(
                choices_indexes
            )
            items_features_by_choice = self._get_items_features_by_choice(
                choices_indexes
            )

            if self.dataset.available_items_by_choice is None:
                available_items_by_choice = np.ones(
                    (len(choices_indexes), self.dataset.base_num_items)
                ).astype("float32")
            else:
                if isinstance(self.dataset.available_items_by_choice, tuple):
                    available_items_by_choice = self.dataset.available_items_by_choice[
                        0
                    ].batch[self.dataset.available_items_by_choice[1][choices_indexes]]
                else:
                    available_items_by_choice = self.dataset.available_items_by_choice[
                        choices_indexes
                    ]
                available_items_by_choice = available_items_by_choice.astype(
                    self.dataset._return_types[2]
                )

            choices = self.dataset.choices[choices_indexes].astype(
                self.dataset._return_types[3]
            )

            if shared_features_by_choice is not None:
                for i in range(len(shared_features_by_choice)):
                    shared_features_by_choice[i] = shared_features_by_choice[i].astype(
                        self.dataset._return_types[0][i]
                    )
                if not self.dataset._return_shared_features_by_choice_tuple:
                    shared_features_by_choice = shared_features_by_choice[0]
                else:
                    shared_features_by_choice = tuple(shared_features_by_choice)

            if items_features_by_choice is not None:
                for i in range(len(items_features_by_choice)):
                    items_features_by_choice[i] = items_features_by_choice[i].astype(
                        self.dataset._return_types[1][i]
                    )
                if not self.dataset._return_items_features_by_choice_tuple:
                    items_features_by_choice = items_features_by_choice[0]
                else:
                    items_features_by_choice = tuple(items_features_by_choice)
            return (
                shared_features_by_choice,
                items_features_by_choice,
                available_items_by_choice,
                choices,
            )

        if isinstance(choices_indexes, slice):
            return self.__getitem__(
                list(range(*choices_indexes.indices(self.dataset.choices.shape[0])))
            )

        if isinstance(choices_indexes, int):
            choices_indexes = [choices_indexes]
            (
                shared_features_by_choices,
                items_features_by_choice,
                available_items_by_choice,
                choice,
            ) = self.__getitem__(choices_indexes)
            if shared_features_by_choices is not None:
                if isinstance(shared_features_by_choices, tuple):
                    shared_features_by_choices = tuple(
                        feat[0] for feat in shared_features_by_choices
                    )
                else:
                    shared_features_by_choices = shared_features_by_choices[0]
            if items_features_by_choice is not None:
                if isinstance(items_features_by_choice, tuple):
                    items_features_by_choice = tuple(
                        feat[0] for feat in items_features_by_choice
                    )
                else:
                    items_features_by_choice = items_features_by_choice[0]

            return (
                shared_features_by_choices,
                items_features_by_choice,
                available_items_by_choice[0],
                choice[0],
            )
        raise NotImplementedError(f"Type{type(choices_indexes)} not handled")

    def get_full_dataset(self, sample_weight=None):
        if self.dataset.shared_features_by_choice is not None:
            shared_features_by_choice = [
                feat for feat in self.dataset.shared_features_by_choice
            ]
        else:
            shared_features_by_choice = None

        if self.dataset.items_features_by_choice is not None:
            items_features_by_choice = [
                feat for feat in self.dataset.items_features_by_choice
            ]
        else:
            items_features_by_choice = None

        if self.dataset.available_items_by_choice is None:
            available_items_by_choice = np.ones(
                (len(self.dataset), self.dataset.base_num_items)
            ).astype("float32")
        else:
            if isinstance(self.dataset.available_items_by_choice, tuple):
                available_items_by_choice = self.dataset.available_items_by_choice[
                    0
                ].batch[self.dataset.available_items_by_choice[1]]
            else:
                available_items_by_choice = self.dataset.available_items_by_choice
        available_items_by_choice = available_items_by_choice.astype(
            self.dataset._return_types[2]
        )

        choices = self.dataset.choices.astype(self.dataset._return_types[3])

        if shared_features_by_choice is not None:
            for i in range(len(shared_features_by_choice)):
                shared_features_by_choice[i] = shared_features_by_choice[i].astype(
                    self.dataset._return_types[0][i]
                )
            if not self.dataset._return_shared_features_by_choice_tuple:
                shared_features_by_choice = shared_features_by_choice[0]
            else:
                shared_features_by_choice = tuple(shared_features_by_choice)

        if items_features_by_choice is not None:
            for i in range(len(items_features_by_choice)):
                items_features_by_choice[i] = items_features_by_choice[i].astype(
                    self.dataset._return_types[1][i]
                )
            if not self.dataset._return_items_features_by_choice_tuple:
                items_features_by_choice = items_features_by_choice[0]
            else:
                items_features_by_choice = tuple(items_features_by_choice)

        if sample_weight is not None:
            return (
                shared_features_by_choice,
                items_features_by_choice,
                available_items_by_choice,
                choices,
            ), sample_weight
        return (
            shared_features_by_choice,
            items_features_by_choice,
            available_items_by_choice,
            choices,
        )
