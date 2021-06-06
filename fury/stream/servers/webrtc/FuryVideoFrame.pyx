from libc.stdint cimport uint8_t
from av.video.frame import VideoFrame


# def copy_array_to_plane(buffer, plane, size_t bytes_per_pixel):
#     cdef size_t i_size = plane.height * plane.width * bytes_per_pixel
#     cdef const uint8_t[:] i_buf = buffer
#     cdef uint8_t[:] o_buf = plane
#     print("%d %d %d"%(i_buf[0],i_buf[1],i_buf[2]))
#     cdef size_t o_stride = abs(plane.line_size)
#     for i in range(i_size):
#         o_buf[i] = i_buf[i]


cdef copy_array_to_plane(buffer, plane, unsigned int bytes_per_pixel):
    cdef const uint8_t[:] i_buf = buffer
    cdef size_t height = plane.height
    cdef size_t width = plane.width
    cdef size_t i_stride = width * bytes_per_pixel
    cdef size_t i_size = height * i_stride
    cdef size_t i_pos = 0

    cdef uint8_t[:] o_buf = plane
    cdef size_t o_pos = 0
    cdef size_t o_stride = abs(plane.line_size)

    while i_pos < i_size:
        o_buf[o_pos:o_pos + i_stride] = i_buf[(i_size-i_pos-i_stride):(i_size-i_pos-i_stride)+i_stride]
        i_pos += i_stride
        o_pos += o_stride

class FuryVideoFrame(VideoFrame):
    def update_from_ndarray(self,array,components=3):
        copy_array_to_plane(array, self.planes[0], components)

