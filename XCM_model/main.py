import os

os.environ["CUDA_VISIBLE_DEVICES"]= "2"    #"1,2,3"

import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from utils.data_loading_2 import import_data
from utils.explainability import get_heatmap
from utils.helpers import create_logger, save_experiment

from models.mtex_cnn import mtex_cnn
from models.xcm import xcm
from models.xcm_seq import xcm_seq

if __name__ == "__main__":

    # Load configuration
    parser = argparse.ArgumentParser(description="XCM")
    parser.add_argument(
        "-c", "--config", default="configuration/config.yml", help="Configuration File"
    )
    parser.add_argument(
        '--fase', type=str, default="N3",
                    help='fase del sueño PSG'
    )
    args = parser.parse_args()
    with open(args.config, "rb") as config_file:
        configuration = yaml.safe_load(config_file)
        

    if configuration["model_name"] in ["XCM", "XCM-Seq"]:
        window_size = configuration["window_size"]
    else:
        window_size = 0
    model_dict = {"XCM": xcm, "XCM-Seq": xcm_seq, "MTEX-CNN": mtex_cnn}

    # Create experiment folder
    xp_dir = (
        "./results/"
        + str(configuration["dataset"])
        + "/"
        + str(configuration["model_name"])
        + "/XP_"
        + str(configuration["experiment_run"])
        + "/"
    )
    save_experiment(xp_dir, args.config)
    log, logclose = create_logger(log_filename=os.path.join(xp_dir, "experiment.log"))
    log("Model: " + configuration["model_name"])

    # Load dataset
    (
        X_train_fold,
        y_train_fold,
        X_test_fold,
        y_test_fold,
        y_train_nonencoded_fold,
        y_test_nonencoded_fold,
    ) = import_data(configuration["dataset"], args.fase, log)
    #print("X_train.shape: ", X_train.shape)
    #print("X_test.shape: ", X_test.shape)

    n_folds = 5

    for i in range(n_folds):
        #take the dataset corresponding to the fold
        X_train = X_train_fold[i]
        y_train = y_train_fold[i]
        X_test = X_test_fold[i]
        y_test = y_test_fold[i]
        y_train_nonencoded = y_train_nonencoded_fold[i]
        y_test_nonencoded = y_test_nonencoded_fold[i]  

        # Instantiate the cross validator
        skf = StratifiedKFold(
            n_splits=configuration["cv_folds"],
            random_state=configuration["random_state"],
            shuffle=True,
        )

        # Instantiate the result dataframes
        train_val_epochs_accuracies = pd.DataFrame(
            columns=["Fold", "Epoch", "Accuracy_Train", "Accuracy_Validation"]
        )
        results = pd.DataFrame(
            columns=[
                "Dataset",
                "Model_Name",
                "Batch_Size",
                "Window_Size",
                "Fold",
                "Accuracy_Train",
                "Accuracy_Validation",
                "Accuracy_Test",
                "F1 Train",
                "F1 Test",
                'F1 Validation'
            ]
        )

        # Loop through the indices the split() method returns
        for index, (train_indices, val_indices) in enumerate(
            skf.split(X_train, y_train_nonencoded)
        ):
            log("\nTraining on fold " + str(index + 1))

            # Generate batches from indices
            xtrain, xval = X_train[train_indices], X_train[val_indices]
            ytrain, yval, ytrain_nonencoded, yval_nonencoded = (
                y_train[train_indices],
                y_train[val_indices],
                y_train_nonencoded[train_indices],
                y_train_nonencoded[val_indices],
            )

            # Train the model
            if configuration["model_name"] in ["XCM", "XCM-Seq"]:
                model = model_dict[configuration["model_name"]](
                    input_shape=X_train.shape[1:],
                    n_class=y_train.shape[1],
                    window_size=configuration["window_size"],
                )
            else:
                model = model_dict[configuration["model_name"]](
                    input_shape=X_train.shape[1:], n_class=y_train.shape[1]
                )
            
            model.compile(
                optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
            )

            h = model.fit(
                xtrain,
                ytrain,
                epochs=configuration["epochs"],
                batch_size=configuration["batch_size"],
                verbose=1,
                validation_data=(xval, yval),
            )

            # Calculate accuracies
            fold_epochs_accuracies = np.concatenate(
                (
                    pd.DataFrame(np.repeat(index + 1, configuration["epochs"])),
                    pd.DataFrame(range(1, configuration["epochs"] + 1)),
                    pd.DataFrame(h.history["accuracy"]),
                    pd.DataFrame(h.history["val_accuracy"]),
                ),
                axis=1,
            )
            acc_train = accuracy_score(
                ytrain_nonencoded, np.argmax(model.predict(xtrain), axis=1)
            )
            f1_train = f1_score(
                ytrain_nonencoded, np.argmax(model.predict(xtrain), axis=1)
            )
            acc_val = accuracy_score(
                yval_nonencoded, np.argmax(model.predict(xval), axis=1)
            )
             
            f1_val = f1_score(
                yval_nonencoded, np.argmax(model.predict(xval), axis=1)
            )
            acc_test = accuracy_score(
                y_test_nonencoded, np.argmax(model.predict(X_test), axis=1)
            )
            f1_test = f1_score(
                y_test_nonencoded, np.argmax(model.predict(X_test), axis=1)
            )

            # Add fold results to the dedicated dataframe
            train_val_epochs_accuracies = pd.concat(
                [
                    train_val_epochs_accuracies,
                    pd.DataFrame(
                        fold_epochs_accuracies,
                        columns=["Fold", "Epoch", "Accuracy_Train", "Accuracy_Validation"],
                    ),
                ],
                axis=0,
            )
            results.loc[index] = [
                configuration["dataset"],
                configuration["model_name"],
                configuration["batch_size"],
                int(configuration["window_size"] * 100),
                index + 1,
                acc_train,
                acc_val,
                acc_test,
                f1_train,
                f1_test,
                f1_val
            ]
            log("Accuracy Test: {0}".format(acc_test))
            log("Accuracy Train: {0}".format(acc_train))
            log("F1 Test: {0}".format(f1_test))
            log("F1 Train: {0}".format(f1_train))

        # Train the model on the full train set
        print('=====Fold {} ======='.format(i))
        
        # Train the model on the full train set
        log("\nTraining on the full train set")
        if configuration["model_name"] in ["XCM", "XCM-Seq"]:
            model = model_dict[configuration["model_name"]](
                input_shape=X_train.shape[1:],
                n_class=y_train.shape[1],
                window_size=configuration["window_size"],
            )
        else:
            model = model_dict[configuration["model_name"]](
                input_shape=X_train.shape[1:], n_class=y_train.shape[1]
            )
        print(model.summary())
        
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(
            X_train,
            y_train,
            epochs=configuration["epochs"],
            batch_size=configuration["batch_size"],
            verbose=1,
        )
        print('00000000000000000000000000000')
        print(model.predict(X_test).shape)
        print(model.predict(X_test)[:1000])
        print(y_test_nonencoded[:1000])

        # Add result to the results dataframe
        acc_test = accuracy_score(
            y_test_nonencoded, np.argmax(model.predict(X_test), axis=1)
        )
        results["Accuracy_Test_Full_Train"] = acc_test
        log("Accuracy Test: {0}".format(acc_test))

        f1_test = f1_score(
            y_test_nonencoded, np.argmax(model.predict(X_test), axis=1)
        )
        results["f1_Test_Full_Train"] = f1_test
        log("f1 Test: {0}".format(f1_test))

        # Export model and results
        model.save(xp_dir + "/model_Fold{}.h5".format(i))
        train_val_epochs_accuracies.to_csv(
            xp_dir + "/train_val_accuracies_Fold{}.csv".format(i), index=False
        )
        results.to_csv(xp_dir + "/results_Fold{}.csv".format(i), index=False)
        print(results)

        # Example of a heatmap from Grad-CAM for the first MTS of the test set
        '''
        get_heatmap(
            configuration,
            xp_dir,
            model,
            X_train,
            X_test,
            y_train_nonencoded,
            y_test_nonencoded,
        )
        '''
        #Guardar las positive y las false rate
        print(y_test_nonencoded.shape)
        #print(model.predict(X_test))
        fpr, tpr, th = roc_curve(y_test_nonencoded, model.predict(X_test)[:,1])
        print(fpr.shape)
        print(tpr.shape)
        #print(th)
        np.save('TruePR_fold{}'.format(i) + args.fase + '.npy', tpr)
        np.save('FalsePR_fold{}'.format(i) + args.fase + '.npy', fpr)
        auc = roc_auc_score(y_test_nonencoded, model.predict(X_test)[:,1])
        print(auc.shape)
        np.save('AUC_fold{}'.format(i) + args.fase + '.npy', np.array(auc))

    logclose()
