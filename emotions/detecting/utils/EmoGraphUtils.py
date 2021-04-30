import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emotions.detecting.Constants import EMOTIONAL_STRESS_POINTS
from emotions.detecting.logs.Logger import Logger
from emotions.detecting.utils import ArrayUtils


def get_irritation_graphs(emotion_events):
    Logger.print("Инициализация параметров графиков")
    initial_irritation_points, middle_irritation_points, final_irritation_points, bar_labels \
        = _create_bar_irritation_params(emotion_events)
    plot_irritation_points, plot_labels = _create_plot_irritation_params(emotion_events)

    data = {"Вопрос задан": initial_irritation_points,
            "Процесс ответа и раздумий": middle_irritation_points,
            "Завершение ответа": final_irritation_points}
    _create_irritation_bar(data, bar_labels, len(initial_irritation_points))
    _create_irritation_plot(plot_irritation_points, plot_labels)


def _create_plot_irritation_params(emotion_events):
    plot_labels = [0.0]
    plot_irritation_points = [0.0]
    index = 0
    coefficient = 1 / (len(emotion_events) * EMOTIONAL_STRESS_POINTS) * 100
    for event in emotion_events:
        index += 1
        _fill_progress_percents_labels(plot_labels, index * EMOTIONAL_STRESS_POINTS, coefficient)
        plot_irritation_points += [event.initial_irritation, event.middle_irritation, event.final_irritation]
    plot_labels += [100.0]
    return plot_irritation_points, plot_labels


def _fill_progress_percents_labels(plot_labels, target_size, coefficient):
    last_label = ArrayUtils.last(plot_labels)
    index = len(plot_labels)
    while index < target_size:
        plot_labels += [last_label + coefficient]
        last_label = ArrayUtils.last(plot_labels)
        index += 1


def _create_bar_irritation_params(emotion_events):
    initial_irritation_points = []
    middle_irritation_points = []
    final_irritation_points = []
    bar_labels = []

    for event in emotion_events:
        initial_irritation_points += [event.initial_irritation]
        middle_irritation_points += [event.middle_irritation]
        final_irritation_points += [event.final_irritation]
        bar_labels += [event.name]
    return initial_irritation_points, middle_irritation_points, final_irritation_points, bar_labels


def _create_irritation_bar(data, labels, length):
    index = np.arange(length)
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.xticks(index, labels)
    plt.title('Уровень эмоционального стресса', fontsize=20)
    plt.xlabel('Вопросы')
    plt.ylabel('Уровень эмоц. стресса, %')
    Logger.print("Отрисовка диаграммы эмоционального стресса")
    plt.show()


def _create_irritation_plot(all_irritation_points, labels):
    df = pd.Series(all_irritation_points, index=labels)
    df.plot.line()
    Logger.print("Отрисовка графика эмоционального стресса")
    plt.title('Уровень эмоционального стресса', fontsize=20)
    plt.xlabel('Прогресс собеседования, %')
    plt.ylabel('Уровень эмоц. стресса, %')
    plt.show()
