import gradio as gr

def odczyt_lini(file_name, line_number):
    with open(file_name, "r") as file:
        line = file.readlines()[line_number - 1]
    return line

def czytaj_tekst_pliku(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def informacje_o_pliku(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    number_lines = len(lines)
    text_lines = czytaj_tekst_pliku(file_name)
    info = f"Plik zawiera {number_lines} linii."
    classes = {}
    for line in text_lines:
        class_name = line.split()[0]
        if class_name not in classes:
            classes[class_name] = 0
        classes[class_name] += 1
    return classes, info, number_lines

def wyswietlanie_pliku(file_name, num_lines):
    classes, info, number_lines = informacje_o_pliku(file_name)
    response = "Liczba klas decyzyjnych: {}\n".format(len(classes))
    num_lines = int(num_lines)
    if num_lines <= 0:
        return "Nieprawidłowe dane wejściowe: liczba mniejsza lub równa zeru."
    elif num_lines > number_lines:
        return "Nieprawidłowe dane wejściowe: liczba wierszy jest większa niż rozmiar pliku."
    line = "Linia poszukiwana: {}\n".format(odczyt_lini(file_name, num_lines))
    for class_name, class_size in classes.items():
        response += "Skala klasy {}: {}\n".format(class_name, class_size)
    return f"{info}\n{response}\n{line}"

file_name_input = gr.inputs.Textbox(lines=1, label="Wprowadź nazwę pliku:")
num_lines_input = gr.inputs.Number(default=1, label="Liczba linii do wyświetlenia:")
output_text = gr.outputs.Textbox(label="Podgląd pliku")

iface = gr.Interface(fn=wyswietlanie_pliku, inputs=[file_name_input, num_lines_input], outputs=output_text, title="Chatbot",
                     description="Wprowadź nazwę pliku i liczbę wierszy do wyświetlenia, aby wyświetlić podgląd pliku.")
iface.launch()
