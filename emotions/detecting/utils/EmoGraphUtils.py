import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    plot_irritation_points = []

    for stage in emotion_stages:
        plot_irritation_points += stage.stage_irritations
    total_points = len(plot_irritation_points)
    percent = 1 / total_points * 100
    for point in plot_irritation_points:
        plot_labels += [plot_labels[-1] + percent]
    plot_irritation_points.insert(0, plot_irritation_points[0] / 2)
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


def _create_irritation_bar(data, labels, stages):
    df = pd.DataFrame(data)
    df.plot(kind='bar')
    plt.xticks(np.arange(stages), labels)
    plt.title('Уровень эмоционального стресса', fontsize=20)
    plt.xlabel('Этапы')
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
