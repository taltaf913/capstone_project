import gradio as gr


myControls = {
    "ResultControl":None,
    "Feedback":None,
    "AdditionalInfo":None
}
def predict(imageToProcess):
    #myControls["ResultContro"].value = "No Disease"

    return ["No Disease", "Nothing is required"]

def submitFeedback(a,b):
    return ["User input submitted successfully"]

with gr.Blocks() as app :

    gr.Markdown(
    """
        # AI based plant Disease Detection Application
       
    """
    )
    imageInput = gr.Image()

    controls = []

    myControls["ResultControl"] = gr.Textbox(label='Possible Disease could be ')
    myControls["AdditionalInfo"] = gr.TextArea(label='Additional Info')
    controls.append(myControls["ResultControl"])
    controls.append(myControls["AdditionalInfo"])
    

    predictBtn = gr.Button(value='Predict')
    predictBtn.click(predict, inputs=imageInput, outputs=controls)


    gr.Markdown()
    myControls["Feedback"] = gr.Checkbox(label="Is prediction acceptable?")
    myControls["UserInput"] = gr.Textbox(label='What is the correct classification?')
    feedbackBtn = gr.Button(value='Submit Feedback')
    feedbackBtn.click(submitFeedback, inputs =[myControls["Feedback"], myControls["UserInput"]], outputs=None)




    app.launch()
