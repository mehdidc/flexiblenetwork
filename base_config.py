from hyperopt import hp
import numpy as np

conv = hp.randint("conv", 10) + 5

options = {
	"PARALLELISM": False,
	"PARALLEL_POWER": 5,
    
    "FOLDERS": {
        "DATA": ["../data/mnist", "../data"],
        "ERRORS": ["errors"],
        "LEARNERS": ["learners"], 
        "STATS": ["stats"],
        "INSTANCES": ["instances"],
        "PREDICTIONS": ["predictions"],
        "BEST_HYPERPARAMS": ["best_hyperparams"],
        "PREPROCESSED": ["preprocessed"],
        "FILTERS": ["filters"],
    },
    "ARCHITECTURES": {
        "mlp_from_file":{
            "load_data": "ilc_ext.load_ilc",
            "filename": "fine_tuning_weights.npy",
            "type": "build_layers.mlp_from_file",
            "input_size": (30, 18, 18),
            "output_size": (2,),
            "activations": "tanh",
        },

        "ilc": {
            "load_data": "ilc_ext.load_ilc",
            "build_input_layer": "ilc_ext.build_input_layer",
            "build_output_layer": "ilc_ext.build_output_layer",
        },
          
        "pseudo3dconv":{
            "extend": "ilc",
            "conv": (11, 11),
            "nbprof": 3,
            "type": "build_layers.pseudo3dconv",
            "activations": ["tanh", "tanh"],
            "hyperopt":{
                "conv": (conv, conv),
                "nbprof": hp.randint("nbprof", 7),
            }
        },
        "3dconv_deep":{
            "extend": "ilc",
            "input_dim": 3, "type": "build_layers.conv",
            "convs": [ (10, 6, 6), (7, 5, 5)],
            "strides": [(2, 2, 2), (1, 1, 1)],
            "pools": [ (1, 1, 1), (1, 1, 1)],
            "nbmaps": [1, 1],
            "activations": ["tanh", "tanh", "tanh"],
            "fc_layer_nb": 10,
        },
        "3dconv_9x9x9":{

            "extend": "ilc",
            "input_dim": 3, "type": "build_layers.conv",
            "preprocessing" : ["preprocessing.histogram_equalization", "preprocessing.mean_std_scaler"],
            "convs":   [ (15, 9, 9)],
            "pools":   None,
            "strides": [(1, 1, 1)],
            "nbmaps":  [1],
            "fc_layer_nb": 10,
            "activations": ["tanh", "tanh"],
        },
        "2dconv_7x7": {
            "extend": "ilc",
            "input_dim": 2, "type": "build_layers.conv",

            "preprocessing": [ "ilc_ext.to_2d", 
                               "preprocessing.mean_std_scaler"],
            
            "build_input_layer": "ilc_ext.build_reduced_to_2d_input_layer",
 
            "convs":   [ (7, 5), (7, 5)],
            "strides": [ (1, 1), (1, 1)],
            "pools":   [ (2, 2), (2, 2)],
            "pools":None,
            "nbmaps":  [1, 1],
            "activations": ["tanh", "tanh", "tanh"],
            "fc_layer_nb": 100,

            "hyperopt":{
                "nbmaps":  [ hp.randint("nbmaps", 100)+1 , 1    ],
                "fc_layer_nb": hp.randint("fc_layer_nb", 2000),
            }

        },
        "2dconv_5x5_nopooling": {

            "extend": "ilc",
            "input_dim": 2, "type": "build_layers.conv",
            "convs":   [ (9, 9)],
            "strides": [ (1, 1)],
            "pools":   None,
            "nbmaps":  [ 1     ],
            "fc_layer_nb": 10,
        },
        "3dconv_5x5_nopooling": {

            "extend": "ilc",
            "input_dim": 3, "type": "build_layers.conv",
            "convs":   [(9, 9, 9), (5, 3, 3)],
            "strides": [(3, 3, 3), (1, 1, 1)],
            "pools": None,
            "nbmaps" : [1, 4],
            "fc_layer_nb": 1000,
            "activations": ["tanh", "tanh", "tanh", "tanh", "tanh"],


            "preprocessing": ["preprocessing.mean_std_scaler"],
            
            "hyperopt":{
                "nbmaps" : [1, hp.randint("nbmaps", 256) + 1],
                "fc_layer_nb": hp.randint("fc_layer_nb", 2000),
            }
        },
       "mlp": {
            "extend": "ilc",
            "nbunits": [10], "type": "build_layers.mlp",
            "activations": ["tanh", "tanh"],
            "preprocessing": ["ilc_ext.to_2d", "preprocessing.hough_transform_2d", "preprocessing.min_max_scaler", "preprocessing.histogram_equalization", "preprocessing.mean_std_scaler"],
            "build_input_layer": "ilc_ext.build_hough_2d_input_layer",
            "hyperopt":{
                "nbunits": [hp.randint("nbunits", 50), hp.randint("nbunits2", 100), hp.randint("nbunits3", 200)],
                "activations": ["tanh", "tanh"]
            }
        },

        "nade": {
            "nbunits": [500], "type": "build_layers.mlp",
            "activations": ["relu", "tanh"],
            "input_dim": (500,),
            "output_dim": (2,),
            "load_data": "data_loaders.load_data",

            "hyperopt":{
                "nbunits": [hp.randint("nbunits", 50), hp.randint("nbunits2", 100)],
                "activations": ["tanh", "tanh"],
            }
        },

        "compress": {

            "nbunits": [300], "type": "build_layers.mlp",
            "input_dim": (28,28),
            "output_dim": (1000,),
            "load_data": "data_loaders.load_data_whole",
            "activations": ["relu", "relu", "relu"],
        
            "preprocessing": ["preprocessing.mean_std_scaler"],

 
        },
 
        "mimic": {

            "nbunits": [300], "type": "build_layers.mlp",
            "input_dim": (1000,),
            "output_dim": (10,),
            "load_data": "data_loaders.load_data",
            "activations": ["relu", "relu", "relu"],
        
            "preprocessing": ["preprocessing.mean_std_scaler"],

 
        },
        "mimic_2048": {

            "nbunits": [100], "type": "build_layers.mlp",
            "input_dim": (4096,),
            "output_dim": (10,),
            "load_data": "data_loaders.load_data_whole",
            "activations": ["relu", "relu", "relu"],
        
            "preprocessing": ["preprocessing.mean_std_scaler"],

 
        },

        "mimic_mix": {

            "nbunits": [15, 5], "type": "build_layers.mlp",
            "input_dim": (5096,),
            "output_dim": (10,),
            "load_data": "data_loaders.load_data_whole",
            "activations": ["tanh", "tanh"],
        
            "preprocessing": ["preprocessing.mean_std_scaler"],

 
        },
 
 

        "mnist":{
            "nbunits": [200], "type": "build_layers.mlp",
            "input_dim": (28, 28),
            "output_dim": (10,),
            "load_data": "data_loaders.load_data",
            "activations": ["tanh", "tanh"],
            "preprocessing": ["preprocessing.mean_std_scaler"],
            #"load_data": "load_data_for_testing_",

            "hyperopt" : {
                
                "nbunits": hp.choice("layers", (
                                    [hp.randint("nb0", 1000)],
                                    [hp.randint("nb1", 500), hp.randint("nb2", 300)],
                                    [hp.randint("nb3", 500), hp.randint("nb4", 300), hp.randint("nb5", 200)])),
                "activations": ["relu", "relu", "relu"],
            }
        },
        "lph":{
            "nbunits": [1000],
            "type": "build_layers.mlp",
            "activations": ["tanh", "tanh"],
            "preprocessing": [],
            "input_dim": (18*18,),
            "output_dim": (2,),
            "load_data": "data_loaders.load_data",
        },
 
        "mnist_conv": {
            "input_dim": 2, "type": "build_layers.conv",
            "convs":   [ (5, 5), (5, 5)],
            "strides": [ (1, 1), (1, 1)],
            "pools":   [ (2, 2), (2, 2)],
            "preprocessing": ["preprocessing.mean_std_scaler"],
            "nbmaps":  [1, 1],
            "activations": ["tanh", "tanh", "tanh", "tanh"],
            "fc_layer_nb": [10],
            "input_dim": (28, 28),
            "output_dim": (10,),
            
             "load_data": "data_loaders.load_data",

            "hyperopt": {
                "nbmaps": [hp.randint("conv1", 8) + 1, 2 + hp.randint("conv2", 8)],
                "fc_layer_nb" : hp.randint("fc", 500) + 1,
            }
        },
 
 
    },
    "LEARNERS": {
        "learner1": {
                "alpha": 0.001, "momentum": 0.5, "lambda1": 0.00, "lambda2": 0.000, 
                 "gradient_check": False, "loss_function": "mse",

        #         "drop_out" : {1:0.5, 2:0.5},

                 #"class_weights": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
                "hyperopt":{
                    "alpha": hp.uniform("alpha", 0.001, 0.005),
                  #  "lambda2": hp.uniform("lambda2", 0, 0.0002),
                }
        },

    },

    "ARCHITECTURE": "mnist",

    
    "LEARNER": "learner1",

    "BATCH_TRAINING":{
        "nb_epochs": 500,
        "batch_size": 20,
        "update_per_epoch": 1,
        "error_rate_min": 0.001,
    },
    "LEARNING": {
#        "VALIDATION": {
#             "ratio": 0.2,
#             "trials": 20
#         }
    },
    "LEARNING_CURVES": {
        "STEP": 100
    },
    "CROSS_VALIDATION": {
        "nb_batches": 10,
        "nb_elements_min": 10,
    },
    "TESTING": {

        "learners" : [


"learner_f78e612a30cb5495c5546dcf16bba2c2_1",
"learner_f78e612a30cb5495c5546dcf16bba2c2_2",
"learner_f78e612a30cb5495c5546dcf16bba2c2_3",
"learner_f78e612a30cb5495c5546dcf16bba2c2_4",
"learner_f78e612a30cb5495c5546dcf16bba2c2_5",
"learner_f78e612a30cb5495c5546dcf16bba2c2_6",
"learner_f78e612a30cb5495c5546dcf16bba2c2_7",
"learner_f78e612a30cb5495c5546dcf16bba2c2_8",
"learner_f78e612a30cb5495c5546dcf16bba2c2_9",
],
        "prediction_file": "pred",
        "mode" : "display_target",
        "dataset": "test"
     },
    "SAVE_FILTERS": {
            "filters": [{"layer": 1, "index": 0}]
    },
    
    "PROBLEM": "classification",
#    "RESUME_LEARNER": {
#        "learner": "learner_e1bde17fcde63e2fad506efc2d4e79ee"
#    },
}
