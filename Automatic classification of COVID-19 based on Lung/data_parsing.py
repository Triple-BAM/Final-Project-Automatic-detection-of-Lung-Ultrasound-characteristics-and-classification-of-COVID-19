import pandas as pd
import plotly.graph_objects as go


def parse_dataset(metadata_path):
    """
    Used to extract information about our dataset.
    It iterates over all images and return a DataFrame with
    the data (output parameters) of all video files.
    :param metadata_path: Path to the Excel spreadsheet.
    :return metadata_df_final: DataFrame with relevant features.
    """
    # Reading the Excel file.
    df_metadata = pd.read_excel(metadata_path)
    # Relevant columns from the DataFrame.
    video_name = df_metadata['video_name']
    covid_severity = df_metadata['covid_severity_grade']
    pleural_line_regular = df_metadata['pleural_line_regular']
    consolidation = df_metadata['consolidation']

    # Creating a new DataFrame with only the relevant columns.
    metadata_df_final = pd.concat([video_name, covid_severity, pleural_line_regular, consolidation], axis=1)
    metadata_df_final.columns = ['video_name', 'covid_severity_grade',
                                 'pleural_line_regular', 'consolidation']
    # If NaN shows up, it is replaced with a zero.
    metadata_df_final = metadata_df_final.fillna(0)

    return metadata_df_final


def plot_distribution(pd_series):
    """
    Plots the data distribution for a specific feature using plotly.
    :param pd_series: Column of a certain feature from the DataFrame.
    """
    labels = pd_series.value_counts().index.tolist()
    counts = pd_series.value_counts().values.tolist()

    pie_plot = go.Pie(labels=labels, values=counts, hole=.3)
    fig = go.Figure(data=[pie_plot])
    fig.update_layout(title_text='Distribution for %s' % pd_series.name)

    fig.show()
