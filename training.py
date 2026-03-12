from core.layer import Layer
from core.model import Model
from ml_tools.optimizers import Optimizer, Nesterov_Accelerated_Gradient, Gradient_Descent, RMS_Propagation, ADAM
from ml_tools.initialisations import he_normal
from utils.data import create_normalized_data, process_df_2_output

from utils.logger import Logger

import plotly.graph_objects as go 
import sys

logger = Logger()


def main():

    if len(sys.argv) != 3:
        logger.error("python training.py *path_to_training_dataset* *path_to_testing_dataset*")
        return 1

    train_file: str = sys.argv[1]
    test_file: str = sys.argv[2]
    
    df_train = create_normalized_data(file_path=train_file)
    df_test = create_normalized_data(file_path=test_file)

    l_train = process_df_2_output(df=df_train) 
    l_test = process_df_2_output(df=df_test)

    try:
        model = Model()

        model.create_network([
                Layer(shape=9),
                Layer(shape=16, activation="Sigmoid", initializer="he_normal"),
                Layer(shape=2, activation="Sigmoid", initializer="he_normal")
            ],
            0.0025,
            2
        )

        opti: Optimizer = ADAM(model.fire, model.network, momentum_rate=0.5, velocity_rate=0.5)
        # opti: Optimizer = Nesterov_Accelerated_Gradient(model.fire, model.network, momentum_rate=0.08)


        model.fit(
            optimizer=opti.stochastic,
            ds_train=l_train,
            ds_test=l_test,
            loss="categorical_cross_entropy",
            epochs=500,
            early_stoper=0.03,
            print_training_state=True,
            history_save=True
        )
        model.network.save_network()
    except Exception as exc:
        if str(exc): logger.error(exc)
        return 1

    epoch = [i for i in range(len(model.accuracies.get("training")))]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch, y=model.accuracies.get("testing"), name="accuracies testing", line={'color': 'darkred', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=model.accuracies.get("training"), name="accuracies training", line={'color': 'firebrick', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=model.losses.get("testing"), name="losses testing", line={'color': 'darkslateblue', 'width': 4}))
    fig.add_trace(go.Scatter(x=epoch, y=model.losses.get("training"), name="losses training", line={'color': 'dodgerblue', 'width': 4}))

    fig.update_layout(
        title=dict(
            text="Accuracies and losses throught the epochs"
        )
    )

    fig.write_html("plots/training_recap.html", auto_open=True)
    return 0



if __name__ == "__main__":
    exit(main())
