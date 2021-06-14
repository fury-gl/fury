
## TODO
- [x] Create a buffer manager object
  - For using DRY principle when multiple encondings became availbale
- [] Widget object compatibility with python versions lower than 3.8
  - memory socket communication  
- [x] MJPEG econding
- [] Fix memory release issues with the shared_manager in python >=3.8 inside the Widget object
- [] Create a tutorial about shared_memory and widget object
- [] update the examples with multiprocessing to be compatible with MacOs
  - setting multiprocessing.set_start_method('spawn')
- [] Implement all the interactions: double-click, touch etc
- []  Widget object: OFF-ScreenRendering should be optional 
- [] Fix cleanup on stream client
  - line 201
   ```
   self.showm.iren.RemoveObserver(self._id_observer)
   AttributeError: 'ShowManager' object has no attribute 'iren
   ```