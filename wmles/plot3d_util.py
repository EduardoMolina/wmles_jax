from plot3d import read_plot3D, connectivity_fast
import metis

blocks = read_plot3D('out/a_airfoil.p3d')
print('Finding connectivity')
face_matches, outer_faces = connectivity_fast(blocks)
print(f'Found {len(face_matches)} face matches and {len(outer_faces)} outer faces')