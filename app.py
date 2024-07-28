from shiny import App, ui, reactive, render
from shinywidgets import output_image, download_button
import pandas as pd
import plotly.graph_objects as go
import io

# Define the UI
app_ui = ui.page_fluid(
    ui.panel_title("Gráfico Piper Interactivo"),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("file", "Sube un archivo CSV", accept=".csv"),
            ui.input_select("color_by", "Agrupar por columna", choices=[]),
            download_button("downloadPlot", "Descargar Gráfico")
        ),
        ui.panel_main(
            ui.output_plot("piperPlot")
        )
    )
)

# Define the server logic
def server(input, output, session):
    
    @reactive.Calc
    def data():
        if not input.file():
            return None
        file_info = input.file()
        return pd.read_csv(file_info[0]['datapath'])
    
    @reactive.Calc
    def columns():
        df = data()
        if df is None:
            return []
        return df.columns.tolist()
    
    @reactive.Effect
    def update_color_by():
        session.update(ui.input_select("color_by", choices=columns()))
    
    @render.plot
    def piperPlot():
        df = data()
        if df is None:
            return None
        
        color_by = input.color_by()
        if color_by and color_by in df.columns:
            categories = df[color_by].unique()
            fig = go.Figure()
            for category in categories:
                category_data = df[df[color_by] == category]
                fig.add_trace(go.Scatter(
                    x=category_data['Ca'], y=category_data['Mg'], 
                    mode='markers', name=category))
        else:
            fig = go.Figure(go.Scatter(
                x=df['Ca'], y=df['Mg'], mode='markers'))
        
        fig.update_layout(title="Piper Diagram", xaxis_title="Ca", yaxis_title="Mg")
        return fig
    
    @session.download
    def downloadPlot():
        df = data()
        if df is None:
            return None
        
        fig = piperPlot()
        fig.write_image("/tmp/piper_diagram.jpg")
        return "/tmp/piper_diagram.jpg", "Piper_plot.jpg", "image/jpeg"
    
    output.piperPlot = piperPlot

app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
