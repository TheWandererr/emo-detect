import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interpolate

from emotions.detecting.Constants import EMOTIONAL_STRESS_POINTS
from emotions.detecting.logs.Logger import Logger


def get_irritation_graphs(emotion_stages):
    Logger.print("Инициализация параметров графиков")
    initial_irritation_points, middle_irritation_points, final_irritation_points, bar_labels \
        = _create_bar_irritation_params(emotion_stages)
    plot_irritation_points, plot_labels = _create_plot_irritation_params(emotion_stages)

    data = {"Вопрос задан": initial_irritation_points,
            "Процесс размышления": middle_irritation_points,
            "Завершение ответа": final_irritation_points}
    _create_irritation_bar(data, bar_labels, len(emotion_stages))
    _create_irritation_plot(plot_irritation_points, plot_labels)


def _create_plot_irritation_params(emotion_stages):
    plot_labels = [0.0]
    plot_irritation_points = [0.0]

    total_points = len(emotion_stages * EMOTIONAL_STRESS_POINTS)
    percent = 1 / total_points * 100

    for stage in emotion_stages:
        plot_irritation_points += [stage.initial_irritation, stage.middle_irritation, stage.final_irritation]
        for index in range(EMOTIONAL_STRESS_POINTS):
            plot_labels += [plot_labels[-1] + percent]

    return plot_irritation_points, plot_labels


def _create_bar_irritation_params(emotion_stages):
    initial_irritation_points = []
    middle_irritation_points = []
    final_irritation_points = []
    bar_labels = []

    for stage in emotion_stages:
        initial_irritation_points += [stage.initial_irritation]
        middle_irritation_points += [stage.middle_irritation]
        final_irritation_points += [stage.final_irritation]
        bar_labels += [stage.name]
    return initial_irritation_points, middle_irritation_points, final_irritation_points, bar_labels


def _create_irritation_bar(data, labels, length):
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.xticks(np.arange(length), labels)
    plt.title('Уровень эмоционального стресса', fontsize=20)
    plt.xlabel('Этапы')
    plt.ylabel('Уровень эмоц. стресса, %')
    Logger.print("Отрисовка диаграммы эмоционального стресса")
    plt.show()


def _create_irritation_plot(all_irritation_points, labels):
    x = np.linspace(labels[0], labels[-1], len(all_irritation_points) * 10)
    spline = interpolate.make_interp_spline(labels, all_irritation_points)
    y = spline(x)
    plt.plot(x, y)
    plt.title('Уровень эмоционального стресса', fontsize=20)
    plt.xlabel('Прогресс собеседования, %')
    plt.ylabel('Уровень эмоц. стресса, %')
    Logger.print("Отрисовка графика эмоционального стресса")
    plt.show()
