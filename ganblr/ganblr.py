from .kdb import *
from .utils import *
import numpy as np
import tensorflow as tf

class GANBLR(object):
    def __init__(self, input_dim, output_dim, constraint) -> None:
        
        KL_LOSS = 0
        _loss = lambda y_true, y_pred: tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)+ KL_LOSS

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(2, input_dim=167, activation='softmax',kernel_constraint=constraint))
        model.compile(loss=_loss, optimizer='adam', metrics=['accuracy'])
        #model.set_weights(weights)

    def train_on_batch(self) -> float:
        pass
    
    def fit(self, x, y, batch_size, epochs, ):
        syn_data,weights = self.warmup_run(x, y, train_idxs, test_idxs, data_name,i+1)
        for i in range(100):
            #loss = kl(X,syn_data[0]).numpy()
            sample_real = pd.DataFrame(df1[train_idxs],columns=df.columns)
            sample_real['label'] = 1
            syn_data = pd.DataFrame(syn_data,columns=df.columns)
            syn_data = syn_data.sample(n=len(sample_real))
            syn_data['label'] = 0
            disc_input = pd.concat([sample_real,syn_data])
            disc_input = disc_input.sample(frac=0.8)
            disc = discrim((sample_real.shape[1]-1),1)
            disc.fit(disc_input.values[:,:-1],disc_input.values[:,-1], batch_size=batch_size,epochs=1)
            prob_fake = disc.predict(sample_real.values[:,:-1])
            ls = np.mean(-np.log(np.subtract(1,prob_fake)))
            syn_data,weights = run_generator(X, y, train_idxs, test_idxs, data_name,i+1,loss=ls,weights=weights)
        pass

    def warmup_run(self, epochs, batch_size):
        ohe_X = OneHotEncoder().fit_transform(X).todense()
        train_data = ohe_X[train_idxs], y[train_idxs]
        test_data  = ohe_X[test_idxs] , y[test_idxs]
        feature_uniques = [len(np.unique(X[:,i])) for i in range(X.shape[1])]
        class_unique = len(np.unique(y))
        logs = []
        
        clear_session()
        constraint = softmax_weight(feature_uniques)
        elr = get_lr(ohe_X.shape[1], class_unique,constraint)
        log_elr = elr.fit(*train_data, validation_data=test_data, batch_size=batch_size,epochs=10)
        logs.append(dict(
            batch_size=batch_size,
            epochs=epochs,
            model='elr',
            data=log_elr.history
        ))
    
        weights_all = elr.get_weights()
        clear_session()
        weights = elr.get_weights()[0]
        _, y_counts = np.unique(y[train_idxs], return_counts=True)
        syn_data = sample_synthetic_data(weights, feature_uniques, y_counts,ohe=False)

    def run_generator(self):
        pass

    def run_discriminator(self):
        pass
    def sample_syenthetic(self, weights, kdb_high_order_encoder, y_counts, split=True, ohe=True, size=None, noise=0) -> tuple(np.ndarray, np.ndarray):
        from pgmpy.models import BayesianModel
        from pgmpy.sampling import BayesianModelSampling
        from pgmpy.factors.discrete import TabularCPD
        #basic varibles
        feature_cards = np.array(kdb_high_order_encoder.feature_uniques_)
        n_features = len(feature_cards)
        n_classes = weights.shape[1]
        n_samples = y_counts.sum()
    
        sample_size = n_samples if size is None else size
        prob_noise = noise
        
        #ensure sum of each constraint group equals to 1, then re concat the probs
        _idxs = np.cumsum([0] + kdb_high_order_encoder.constraints_.tolist())
        constraint_idxs = [(_idxs[i],_idxs[i+1]) for i in range(len(_idxs)-1)]
        
        probs = np.exp(weights)
        cpd_probs = [probs[start:end,:] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p/p.sum(axis=0) for p in cpd_probs])
    
        #assign the probs to the full cpd tables
        idxs = np.cumsum([0] + kdb_high_order_encoder.high_order_feature_uniques_)
        feature_idxs = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
        have_value_idxs = kdb_high_order_encoder.have_value_idxs_
        full_cpd_probs = [] 
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            #(n_high_order_feature_uniques, n_classes)
            cpd_prob_ = cpd_probs[start:end,:]
            #(n_all_combination) Note: the order is (*parent, variable)
            have_value_ravel = have_value.ravel()
            #(n_classes * n_all_combination)
            have_value_ravel_repeat = np.hstack([have_value_ravel] * n_classes)
            #(n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
            full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
            full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
            #(n_classes * n_parent_combinations, n_variable_unique)
            full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
            full_cpd_prob = _add_uniform(full_cpd_prob, noise=prob_noise)
            full_cpd_probs.append(full_cpd_prob)
    
        #prepare node and edge names
        node_names = [str(i) for i in range(n_features + 1)]
        edge_names = [(str(i), str(j)) for i,j in kdb_high_order_encoder.edges_]
        y_name = node_names[-1]
    
        #create TabularCPD objects
        evidences = kdb_high_order_encoder.dependencies_
        feature_cpds = [
            TabularCPD(str(name), feature_cards[name], table, 
                       evidence=[y_name, *[str(e) for e in evidences]], 
                       evidence_card=[n_classes, *feature_cards[evidences].tolist()])
            for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
        ]
        y_probs = (y_counts/n_samples).reshape(-1,1)
        y_cpd = TabularCPD(y_name, n_classes, y_probs)
    
        #create kDB model, then sample data
        model = BayesianModel(edge_names)
        model.add_cpds(y_cpd, *feature_cpds)
        result = BayesianModelSampling(model).forward_sample(size=sample_size)
        sorted_result = result[node_names].values
    
        #return
        if not split:
            return sorted_result
            
        syn_X, syn_y = sorted_result[:,:-1], sorted_result[:,-1]
        if ohe:
            from sklearn.preprocessing import OneHotEncoder
            ohe_syn_X = OneHotEncoder().fit_transform(syn_X)
            return ohe_syn_X, syn_y
        else:
            return syn_X, syn_y    
            pass