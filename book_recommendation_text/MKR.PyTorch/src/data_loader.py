import numpy as np
import os

class RSDataset:
    def __init__(self, args):
        self.args = args
        self.n_user, self.n_item, self.raw_data, self.data, self.indices = self._load_rating()

    def __getitem__(self, index):
        return self.raw_data[index]

    def _load_rating(self):
        print('Reading rating file')

        rating_file = os.path.join('..', 'data', self.args.dataset, 'ratings_final')
        if os.path.exists(rating_file + '.npy'):
            rating_np = np.load(rating_file + '.npy')
        else:
            rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int32)
            np.save(rating_file + '.npy', rating_np)

        n_user = len(set(rating_np[:, 0]))
        n_item = len(set(rating_np[:, 1]))
        raw_data, data, indices = self._dataset_split(rating_np)

        return n_user, n_item, raw_data, data, indices


    def _dataset_split(self, rating_np):
        print('Splitting dataset')

        # train:eval:test = 6:2:2
        eval_ratio = 0.2
        test_ratio = 0.2
        n_ratings = rating_np.shape[0]

        eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
        left = set(range(n_ratings)) - set(eval_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))

        train_data = rating_np[train_indices]
        eval_data = rating_np[eval_indices]
        test_data = rating_np[test_indices]

        return rating_np, [train_data, eval_data, test_data], [train_indices, eval_indices, test_indices]

class KGDataset:
    def __init__(self, args):
        self.args = args
        self.n_entity, self.n_relation, self.kg = self._load_kg()

    def __getitem__(self, index):
        return self.kg[index]

    def __len__(self):
        return len(self.kg)

    def _load_kg(self):
        print('Reading KG file')

        kg_file = os.path.join('..', 'data', self.args.dataset, 'kg_final')
        if os.path.exists(kg_file + '.npy'):
            kg = np.load(kg_file + '.npy')
        else:
            kg = np.loadtxt(kg_file + '.txt', dtype=np.int32)
            np.save(kg_file + '.npy', kg)

        n_entity = len(set(kg[:, 0]) | set(kg[:, 2]))
        n_relation = len(set(kg[:, 1]))

        return n_entity, n_relation, kg

