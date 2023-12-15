from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from solver import Solver
from PIL import Image, ImageTk
from threading import Thread

class GUI:
    def __init__(self, master, figure, ax):
        self.im = Image.open("C:\\workspace\\math_modeling_final\\resources\\exploded.png")
        
        self.master = master
        self.master.title("Hodgkin Huxley Model")
        self.figure = figure
        self.ax : Axes = ax
        self.Scales = None
        self.create_widgets()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.image_canvas = ImageTk.PhotoImage(self.im)
        self.image_label = Label(image=self.image_canvas)
        self.error = False
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.solver = Solver()
        
        Tk.report_callback_exception = self.error_screen
        
        self.update_canvas()     

    
    def error_screen(self, e, _, _2):
        print(e)
        self.canvas.get_tk_widget().pack_forget()
        self.image_label.pack(side=TOP, fill=BOTH, expand=1)
        self.error = True
        self.master.after(100, self.update_canvas)

    def create_widgets(self):
        # Create a button to trigger the Matplotlib plot
        self.Scales = Frame(self.master)
        self.T_scale = Scale(self.Scales, variable=DoubleVar(value=25), orient=VERTICAL, cursor="dot", label="Max Time", resolution=0.01, from_=100, to=-5, name="t")
        self.T_scale.grid(row=1, column=1)
        I_scale = Scale(self.Scales, variable=DoubleVar(value=50), orient=VERTICAL, cursor="dot", label="I_in", resolution=0.01, from_=100, to=0, name="i")
        I_scale.grid(row=1, column=2)
        gk_scale = Scale(self.Scales, variable=DoubleVar(value=36), orient=VERTICAL, cursor="dot", label="g_K", resolution=0.01, from_=50, to=-50, name="gk")
        gk_scale.grid(row=1, column=3)
        gna_scale = Scale(self.Scales, variable=DoubleVar(value=120), orient=VERTICAL, cursor="dot", label="g_NA", resolution=0.01, from_=200, to=-200, name='gna')
        gna_scale.grid(row=1, column=4)
        gl_scale = Scale(self.Scales, variable=DoubleVar(value=0.3), orient=VERTICAL, cursor="dot", label="g_leak", resolution=0.01, from_=50, to=-50, name = 'gl')
        gl_scale.grid(row=1, column=5)
        vk_scale = Scale(self.Scales, variable=DoubleVar(value=-12), orient=VERTICAL, cursor="dot", label="v_K", resolution=0.01, from_=50, to=-50, name = 'vk')
        vk_scale.grid(row=1, column=6)
        vna_scale = Scale(self.Scales, variable=DoubleVar(value=115), orient=VERTICAL, cursor="dot", label="v_NA", resolution=0.01, from_=200, to=-200, name='vna')
        vna_scale.grid(row=1, column=7)
        vl_scale = Scale(self.Scales, variable=DoubleVar(value=10.6), orient=VERTICAL, cursor="dot", label="v_leak", resolution=0.01, from_=50, to=-50, name='vl')
        vl_scale.grid(row=1, column=8) 
        cm_scale = Scale(self.Scales, variable=DoubleVar(value=1), orient=VERTICAL, cursor="dot", label="C_m", resolution=0.01, from_=10, to=-10, name='cm')
        cm_scale.grid(row=1, column=9) 
        self.Scales.pack(side=TOP)
        
        self.Radios = Frame(self.master)
        self.solver_selector = IntVar(value=4)
        Radiobutton(self.Radios, text="Forward Euler",  variable = self.solver_selector, value = 1).grid(row = 0, column = 0)
        Radiobutton(self.Radios, text="RK4", variable = self.solver_selector, value = 2).grid(row = 0, column = 1)
        Radiobutton(self.Radios, text="Adams-Bashforth",  variable = self.solver_selector, value = 3).grid(row = 1, column = 0)
        Radiobutton(self.Radios, text="Numpy",   variable = self.solver_selector, value = 4).grid(row = 1, column = 1)
        self.Radios.pack()
    def create_plot(self):
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    
    def update_canvas(self):
        
        if self.error:
            self.image_label.pack_forget()
            self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
            self.error = False
        params = {}
        for scale in self.Scales.winfo_children():
            name = scale.winfo_name()
            params[name] = scale.get()
        self.solver.set_config(config=params)
        
        if self.solver_selector.get() == 1:
            T, V = self.solver.solve_with_adams_bashforth(tmax=self.T_scale.get(), s=1) # aka forward euler
        elif self.solver_selector.get() == 2:
            T, V = self.solver.solve_with_rk4(tmax=self.T_scale.get())
        elif self.solver_selector.get() == 3:
            T, V = self.solver.solve_with_adams_bashforth(self.T_scale.get())
        elif self.solver_selector.get() == 4:
            T, V = self.solver.solve_with_numpy(self.T_scale.get())
            
        self.ax.clear()
        self.ax.plot(T, V)
        self.canvas.draw()
        
        self.master.after(100, self.update_canvas)
        
if __name__ == "__main__":
    # Example: Create a Matplotlib figure
    example_figure = Figure(figsize=(5, 4), dpi=150)
    ax = example_figure.add_subplot(1, 1, 1)

    # Create the main Tkinter window and MatplotlibTkinterGUI instance
    root = Tk()
    
    
    app = GUI(root, figure=example_figure, ax=ax)

    # Run the Tkinter event loop
    root.mainloop()
