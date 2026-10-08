import numpy as np


def calculate_mape(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    non_zero = actual != 0

    return (
        np.mean(
            np.abs(
                (
                    actual[non_zero]
                    - predicted[non_zero]
                )
                / actual[non_zero]
            )
        )
        * 100
    )


def test_mape():

    actual = [100, 200, 300]

    predicted = [90, 210, 290]

    result = calculate_mape(
        actual,
        predicted
    )

    assert result >= 0


def test_perfect_forecast():

    actual = [100, 200, 300]

    predicted = [100, 200, 300]

    result = calculate_mape(
        actual,
        predicted
    )

    assert result == 0


def test_forecast_values_are_non_negative():

    predictions = np.array(
        [100, 200, 50, 0]
    )

    assert np.all(
        predictions >= 0
    )

