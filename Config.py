import PySpin

def configure_camera(cam):
    cam.Init()

    nodemap = cam.GetNodeMap()

    # -- Set Pixel Format --
    pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
        mono16_entry = pixel_format.GetEntryByName("Mono16")
        if PySpin.IsAvailable(mono16_entry) and PySpin.IsReadable(mono16_entry):
            pixel_format.SetIntValue(mono16_entry.GetValue())
            print("Pixel Format set to Mono16.")

    # -- Set FLIR Measurement Values --
    def set_float(node_name, value):
        node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            node.SetValue(float(value))
            print(f"{node_name} set to {value}")

    def set_enum(node_name, entry_name):
        node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
        if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
            entry = node.GetEntryByName(entry_name)
            if PySpin.IsAvailable(entry) and PySpin.IsReadable(entry):
                node.SetIntValue(entry.GetValue())
                print(f"{node_name} set to {entry_name}")

    set_float("R", 554118)
    set_float("B", 1597.67)
    set_float("F", 1)
    set_float("O", 49750)
    set_float("Transmission", 0.68)
    set_float("FNumber", 1.25)
    set_float("J1", 33.47)

    set_enum("TemperatureLinearMode", "Off")
    set_enum("TemperatureLinearResolution", "High")

    cam.DeInit()

def main():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("No FLIR cameras detected.")
        system.ReleaseInstance()
        return

    for i in range(cam_list.GetSize()):
        cam = cam_list.GetByIndex(i)
        try:
            print(f"\nConfiguring Camera {i + 1}/{cam_list.GetSize()}...")
            configure_camera(cam)
        except Exception as e:
            print(f"Error configuring camera {i + 1}: {e}")

    cam = None
    cam_list.Clear()
    system.ReleaseInstance()



if __name__ == "__main__":
    main()
