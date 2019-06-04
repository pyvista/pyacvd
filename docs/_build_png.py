from pyvista import examples
import pyacvd

import pyvista
pyvista.set_plot_theme('document')

# download cow mesh
cow = examples.download_cow()

cpos = [(15.974333902609903, 8.426371781546546, -17.12964912391155),
        (0.7761263847351074, -0.4386579990386963, 0.0),
        (-0.23846635120392892, 0.9325600395795517, 0.2710453318595791)]


# plot original mesh
cow.plot(show_edges=True, color='w', cpos=cpos,
         screenshot='/home/alex/afrl/python/source/pyacvd/docs/images/cow.png')

cpos = [(7.927519161395299, 3.54223003919585, -4.1077249997544545),
 (2.5251427740425236, 0.3910539874485469, 1.9812043586464985),
 (-0.23846635120392892, 0.9325600395795517, 0.2710453318595791)]

cow.plot(show_edges=True, color='w', cpos=cpos,
         screenshot='/home/alex/afrl/python/source/pyacvd/docs/images/cow_zoom.png')


# mesh is not dense enough for uniform remeshing
# must be an all triangular mesh to sub-divide
cow.tri_filter(inplace=True)
cow.subdivide(4, inplace=True)

clus = pyacvd.Clustering(cow)
clus.cluster(20000)

# plot clustered cow mesh
cpos = [(7.927519161395299, 3.54223003919585, -4.1077249997544545),
 (2.5251427740425236, 0.3910539874485469, 1.9812043586464985),
 (-0.23846635120392892, 0.9325600395795517, 0.2710453318595791)]

clus.plot(screenshot='/home/alex/afrl/python/source/pyacvd/docs/images/cow_clus.png',
          cpos=cpos, cmap='bwr')

# remesh
remesh = clus.create_mesh()

# plot uniformly remeshed cow
remesh.plot(color='w', show_edges=True, cpos=cpos, smooth_shading=True,
            screenshot='/home/alex/afrl/python/source/pyacvd/docs/images/cow_remesh.png')
