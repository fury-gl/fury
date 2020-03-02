import fury.ui as ui
import fury.window as window


def display_element():
    value=values[values.index(listbox.selected[0])]
    print(value);
    textbox.message=value;

values = ['abcdefghijklmnopqrstuvwxyz', 'ahdfjkhkashdfjkhdskfhiweuroqweqr', 'akdiuhdaiufhasdfusdffsdfusdyf', 'Soham']
listbox = ui.ListBox2D(values=values, size=(300, 300))
listbox.on_change=display_element;
panel = ui.Panel2D(size=(300,100),position=(550,250) ,color=(0,0,0), align="center")
panel.center = (300,200)
textbox=ui.TextBlock2D();
panel.add_element(textbox, (150,50))
sm=window.ShowManager(size=(800,800))
sm.scene.add(panel);
sm.scene.add(listbox);
sm.start()


