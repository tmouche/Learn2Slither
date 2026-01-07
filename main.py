import tkinter

def calcul():
    try:
        entree.configure(background='white')
        result.set(eval(expression.get()))
    except:
        entree.configure(background='lightcoral')

window = tkinter.Tk()

titre = tkinter.Label(window, text="Hello World!")
titre.configure(fg='red')
titre.pack()

window.title("Learn2Slither")
window.minsize(300, 200)

cadre = tkinter.Frame(window)
cadre.pack()

expression = tkinter.StringVar()
expression.set("Enter your expression here")
entree = tkinter.Entry(cadre, textvariable=expression, width=30)
entree.pack()

result = tkinter.StringVar()

boutton = tkinter.Button(cadre, text="Compute", command=calcul)
boutton.pack()

output = tkinter.Label(cadre, textvariable=result)
output.pack()



window.mainloop()
