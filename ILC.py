import sys
import os
import gzip
from itertools import chain
from feature_map import FeatureMap, Learner, Layer
import json
import numpy as np
from error_rate import classification_error_rate, where_error, confusion_matrix
from easy import build_weighted_pool_layer, build_convpool_layers, cross_validation
from collections import defaultdict
import config
from constants import *
from utils import ressource_full_filename, with_job_suffix, import_object, save_learner

from dataset import Dataset
from easy import batch_training
from itertools import repeat
from utils import Stats
from utils import ndhash, concat_features, from_hamming, to_hamming

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample

import copy

def instantiate_architecture(archs, name):
    if "extend" in archs[name]:
        name_extension = archs[name]["extend"]
        del archs[name]["extend"]
        orig = copy.copy(archs[name])
        archs[name] = {}
        archs[name].update(instantiate_architecture(archs, name_extension))
        archs[name].update(orig)
    else:
        return archs[name]
    
def process_architectures():
    archs = config.options["ARCHITECTURES"]

    for name in archs.keys():
        instantiate_architecture(archs, name)
   
    return archs


def get_architecture_from_config():
    arch = config.options["ARCHITECTURES"]
    use = config.options["ARCHITECTURE"]
    print use
    arch_to_use = arch[use]
    return arch_to_use

def get_hyperopt_from_config():
    arch_to_use = get_architecture_from_config()
    hyp = {}
    
    hyp["ARCHITECTURE"] = {}
    hyp["ARCHITECTURE"].update(arch_to_use)
    hyp["ARCHITECTURE"].update(arch_to_use["hyperopt"])

    hyp["LEARNER"] = {}
    hyp["LEARNER"].update(get_learner_config()["hyperopt"])
    return hyp

def get_builder_from_architecture(arch):
    #builder = architecture_types[arch["type"]]
    
    print arch["type"]
    builder = import_object(arch["type"])
    return builder
    
def build_layers_from_config():
    arch_to_use = get_architecture_from_config()
    builder = get_builder_from_architecture(arch_to_use)
    print "using architecture %s" % (arch_to_use,)
    return build_layers(builder, arch_to_use)

def build_layers(builder, arch_to_use):
    arch_to_use_ =  {}
    arch_to_use_.update(arch_to_use)
    if "hyperopt" in arch_to_use_:
        del arch_to_use_["hyperopt"]
    return builder(**arch_to_use_)

def load_data_from_config():
    arch = get_architecture_from_config()
    # loaders[arch.get("load_data", "load_ilc")]()
    return import_object(arch.get("load_data", "data_loaders.load_data"))()

def get_preprocessing_chain_from_config():
    arch = get_architecture_from_config()
    chain = arch.get("preprocessing", [])
    return map(lambda o:import_object(o), chain)

    

def get_learner_config():
    options = config.options["LEARNERS"][config.options["LEARNER"]]
    return options

import uuid
import md5
import pprint

from preprocessing import apply_preprocessing_chain

from hyperopt.mongoexp import MongoTrials

def hyperparameters_optimize(hyperparameters):
    print "trying %s" % (hyperparameters,)
    builder = hyperparameters["BUILDER"]
    train_ds = hyperparameters["TRAIN_DS"]
    valid_ds = hyperparameters["TEST_DS"]
    batch_training_options = hyperparameters["BATCH_TRAINING_OPTIONS"]

    layers = build_layers(builder, hyperparameters["ARCHITECTURE"])
    learner = Learner(layers, options=hyperparameters["LEARNER"])
    learner.init()
    layers = copy.deepcopy(learner.layers)
    local_stats = Stats()
    batch_training(learner, train_ds, valid_ds, stats=local_stats, **self.batch_training_options)
    best_test_error_rate = min(map(lambda p:p[1],  local_stats.get_points("test_error_rate")))
    return {"loss": best_test_error_rate, "status" : STATUS_OK, "params": hyperparameters, "layers": layers}


def load_learner(filename):
    fd = gzip.open(ressource_full_filename(with_job_suffix(filename), LEARNERS, output=False), "r")
    learner = Learner.load(fd)
    fd.close()
    return learner


from utils import conv_2d_filter, conv_3d_filter
def ensemble_predict(learners, ds):
    pred_avg = None
    for l in learners:
        output_fmap = l.layers[-1].feature_maps[0]

        test_i, test_o = l.from_dataset_to_data(ds)
        p= l.predict(test_i, test=True)[output_fmap]
        if pred_avg is None:
            pred_avg = p
        else:
            pred_avg += p
    pred_avg /= len(learners)
    return pred_avg

def ILC():

    #if config.options["MODE"] == "TESTING" and "config" in config.options["TESTING"]:
    #    config.options.update(json.load(ressource_full_filename(config.options.ge("TESTING").get("config"), INSTANCES)))


    instance_name = str(md5.md5(str(uuid.uuid4())).hexdigest())
    f = open(ressource_full_filename(instance_name, INSTANCES, output=True), "w")
    pp = pprint.PrettyPrinter(indent=4, stream=f)
    pp.pprint(config.options)
    f.close()

    print "Instance name : %s" % (instance_name,)
    instance_name = "_" + instance_name

    process_architectures()

    train_ds, test_ds = load_data_from_config()
    
    #n = np.arange(0, train_ds.input.shape[0])
    #train_ds.input = train_ds.input[n[n>=0]]
    #train_ds.output = train_ds.output[n[n>=0]]

    preprocessing_chain = get_preprocessing_chain_from_config()
    train_ds.input = apply_preprocessing_chain(train_ds.input, preprocessing_chain)
    test_ds.input = apply_preprocessing_chain(test_ds.input, preprocessing_chain)

    #train_ds.truncate(0000)
    #test_ds.truncate(100)
    learner = None

    if "RESUME_LEARNER" in config.options:
        print "resuming..."
        learner_filename = config.options["RESUME_LEARNER"]["learner"]
        learner = load_learner(learner_filename)
    else:
        learner = None

    
    print "Nb of training examples : %d" % (train_ds.input.shape[0],)
    print "Nb of testing examples : %d" % (test_ds.input.shape[0],)
    layers = build_layers_from_config()

    options = get_learner_config()


    print "using learner %s" % (config.options["LEARNER"])
    
    if learner is None:
        learner = Learner(layers, options=options)
    stats = Stats()

    if config.options["MODE"] == "SAVE_PREPROCESSED":
        np.save(ressource_full_filename("train" + instance_name, PREPROCESSED, output=True), (train_ds.input, train_ds.output))
        np.save(ressource_full_filename("test" + instance_name, PREPROCESSED, output=True), (test_ds.input, test_ds.output))

    elif config.options["MODE"] == "SAVE_FILTERS":
        import matplotlib.pyplot as plt
        from show import show_3d
        folder = ressource_full_filename(instance_name, FILTERS, output=True)
        try:
            os.mkdir(folder)
        except Exception:
            pass
        k = 1
        for d in chain(train_ds.input, test_ds.input):
            for f in config.options["SAVE_FILTERS"].get("filters"):
                layer = f["layer"]
                index = f["index"]
         
                F = learner.layers[layer].feature_maps[index].W.W
                stride = learner.layers[layer].feature_maps[index].W.stride
                if len(d.shape) == 3:
                    plt.subplot(1, 2, 1)
                    show_3d([d])
                    plt.subplot(1, 2, 2)
                    show_3d([F])
                else:
                    plt.subplot(1, 2, 1)
                    plt.imshow(d, cmap='winter')
                    plt.subplot(1, 2, 2)
                    plt.imshow(conv_2d_filter(d, F), cmap='winter')
                plt.savefig(folder + "/layer%d_filter%d_example_%d.png" % (layer, index, k)  )
            k += 1

    elif config.options["MODE"] == "TESTING":
        print" loading the learner..."
        learners = []
        for l in config.options["TESTING"]["learners"]:
            learner = load_learner(l)
            learners.append(learner)
        print "predicting..."

        dataset = config.options["TESTING"].get("dataset", "test")
        if dataset == "train":
            ds = train_ds
        elif dataset == "test":
            ds = test_ds
        else:
            raise Exception("no valid dataset")

        
        pred_avg = ensemble_predict(learners, ds)
        targets =  from_hamming(pred_avg)
        preds = ressource_full_filename(config.options["TESTING"]["prediction_file"] + instance_name, PREDICTIONS, output=True)

        fd = open(preds, "w")
        print len(ds.input)
        
        mode = config.options.get("TESTING").get("mode")
        if mode == "display_idx_errors" or mode == "display_target_real":
            output = from_hamming(ds.output)
        else:
            output = None


        err =  0
         
        for k, target in enumerate(targets):
            if mode == "display_idx_errors":
                if (target) != (real):
                    fd.write("%d\n" % (k,))
                    err += 1
            elif mode == "display_target_real":
                fd.write("%d %d\n" % (target,real))
            elif mode == "display_target":
                fd.write("%d\n" % (target,))

        fd.close()
        if mode == "display_idx_Errors" or mode == "display_target_real":
            print "%d errors / %d = %f" % (err, len(targets), float(err) / len(targets))
    elif config.options["MODE"] == "LEARNING_CURVES":
        #layers = learner.layers
        step = config.options["LEARNING_CURVES"]["STEP"]
        for nbexamples in xrange(1, train_ds.input.shape[0], step):
            print "Nb examples : %d" % (nbexamples,)
            
            learner.init()

            train_ds_ = train_ds.get_random_batch_as_dataset(nbexamples)
            local_stats = Stats()
            batch_training(learner, train_ds_, test_ds, stats=local_stats, **config.options["BATCH_TRAINING"])
            
            best_test_error_rate = min(map(lambda p:p[1],  local_stats.get_points("test_error_rate")) )
            best_train_error_rate = min(map(lambda p:p[1], local_stats.get_points("train_error_rate") ))
            stats.new_point("learning_curve_train", (nbexamples, best_train_error_rate))
            stats.new_point("learning_curve_test", (nbexamples, best_test_error_rate))

    elif config.options["MODE"] == "LEARNING":

        if "VALIDATION" in config.options["LEARNING"]:
            ratio = config.options["LEARNING"]["VALIDATION"]["ratio"]
            train_ds_, validation_ds = train_ds.break_to_datasets((1. - ratio, ratio))

            def fn(hyperparameters):
                print "trying %s" % (hyperparameters,)
                architecture = get_architecture_from_config()
                builder = get_builder_from_architecture(architecture)
                
                layers = build_layers(builder, hyperparameters["ARCHITECTURE"])
                
                learner = Learner(layers, options=hyperparameters["LEARNER"])
                learner.init()
                local_stats = Stats()
                batch_training(learner, train_ds_, validation_ds, stats=local_stats, **config.options["BATCH_TRAINING"])
                layers = copy.deepcopy(learner.layers)
                best_test_error_rate = min(map(lambda p:p[1],  local_stats.get_points("test_error_rate")))
                return {"loss": best_test_error_rate, "status" : STATUS_OK, "params": hyperparameters, "layers": layers}
           
            space = get_hyperopt_from_config()
            
            #trials = MongoTrials('mongo://localhost/hyperopt/jobs', exp_key='exp1')
            trials = Trials()
            fmin(fn=fn, space=space, algo=tpe.suggest, max_evals=config.options["LEARNING"]["VALIDATION"]["trials"], trials=trials)
            best_result = min(trials.trials, key=lambda trial:trial["result"]["loss"])["result"]


            layers = best_result["layers"]


            learners = []
            for trial in trials.trials:
                layers_cur_trial = trial["result"]["layers"]
                learners.append(Learner(layers_cur_trial, options=options))
            

            learner = Learner(layers, options=options)

            o_ensemble = (ensemble_predict(learners, test_ds))
            o_best = ensemble_predict([learner], test_ds)

            print "Best params in validation : %s" % (best_result["params"],)
            f = open(ressource_full_filename("best_hp" + instance_name, BEST_HYPERPARAMS, output=True), "w")
            pp = pprint.PrettyPrinter(indent=4, stream=f)
            pp.pprint(best_result["params"])
            f.close()

            print "Best learner error rate : %f" % (classification_error_rate(test_ds.output, o_best),)
            print "Ensemble error rate : %f" % (classification_error_rate(test_ds.output, o_ensemble),)

            min_test_layers = learner.layers
            #print "learning with this..."
            #stats = Stats()
            #min_train_layers, min_test_layers = batch_training(learner, train_ds, test_ds, stats=stats, **config.options["BATCH_TRAINING"])
        else:
            min_train_layers, min_test_layers = batch_training(learner, train_ds, test_ds, stats=stats, **config.options["BATCH_TRAINING"])

        learner.layers = min_test_layers
        
        train_input, train_output = learner.from_dataset_to_data(train_ds)
        test_input, test_output = learner.from_dataset_to_data(test_ds)
        predicted_output = learner.predict(test_input, test=True)
        output_fmap = learner.layers[-1].feature_maps[0]
        
        test_error_rate = classification_error_rate(test_output[output_fmap], predicted_output[output_fmap])
        print "Final test error rate : %f" % (test_error_rate,)
        print "Final confusion matrix of test : "
        print confusion_matrix(test_output[output_fmap], predicted_output[output_fmap])

        print "saving errors..."
        errors = np.array(list(where_error(test_output[output_fmap], predicted_output[output_fmap])))
        np.savetxt(ressource_full_filename(with_job_suffix("error") + instance_name, ERRORS, output=True), errors, fmt='%d')

        print "saving learner..."
        save_learner(learner, "learner" + instance_name)
        if "SAVE_FEATURES" in config.options["LEARNING"]:
            save_features = config.options["LEARNING"]["SAVE_FEATURES"]
            layer = save_features["until_layer"]
            
            train_features = concat_features(learner.predict(train_input, test=True, until_layer=layer))
            train_output = from_numbers_to_str_targets(concat_features(train_output))

            test_features = concat_features(learner.predict(test_input, test=True, until_layer=layer))
            test_output = from_numbers_to_str_targets(concat_features(test_output))
            print "saving features..."
            np.save(ressource_full_filename("extracted_features/train" + instance_name, DATA, output=True), (train_features, train_output))
            np.save(ressource_full_filename("extracted_features/test" + instance_name, DATA, output=True), (test_features, test_output))



    elif config.options["MODE"] == "CROSS_VALIDATION":
        print "CROSS VALIDATION..."
        ds = train_ds
        stats = Stats()
        min_train_layers, min_test_layers, layers = cross_validation(learner, ds, config.options["CROSS_VALIDATION"]["nb_batches"], config.options["CROSS_VALIDATION"]["nb_elements_min"], stats)
        
        learner.layers = min_test_layers
        print "saving learner..."
        save_learner(learner, "learner" + instance_name)

        learners = []
        for l in layers:
            learners.append(Learner(l))
        o_ensemble = (ensemble_predict(learners, test_ds))
        o_best = (ensemble_predict([learner], test_ds))
        
        
        print "Best learner error rate : %f" % (classification_error_rate(test_ds.output, o_best),)
        print "Ensemble error rate : %f" % (classification_error_rate(test_ds.output, o_ensemble),)

        test_input, test_output = learner.from_dataset_to_data(ds)
        predicted_output = learner.predict(test_input)
        output_fmap = learner.layers[-1].feature_maps[0]

        print "saving errors"
        error_indexes = list(where_error(test_output[output_fmap], predicted_output[output_fmap]))
        error_hashes = [ndhash(ds.input[ind]) for ind in error_indexes]
        errors = np.array([error_indexes, error_hashes]).T
        np.savetxt(ressource_full_filename(with_job_suffix("error") + instance_name, ERRORS, output=True), errors, fmt="%s")

        print "saving learners..."
        for i, l in enumerate(layers):
            learner.layers = l
            save_learner(learner, "learner%s_%d" % (instance_name, i))


    
    print "saving stats"
    for subject in (stats.subjects()):
        a = np.array(stats.get_points(subject))
        np.savetxt(ressource_full_filename(with_job_suffix(subject) + instance_name, STATS, output=True), a)
    
    if learner.workers is not None:
        learner.workers.close()

    print "end of %s" % (instance_name[1:],)

if __name__ == "__main__":
    import time
    s = int(time.time())
    np.random.seed(s)
    print "seed : %d" % (s,)

    a = time.time()
    ILC()
    print "Took %f  seconds" % (time.time() - a,)

