import numpy as np
import cv2
import dlib
from scipy.spatial import Delaunay
from PIL import Image
import streamlit as st
from matplotlib import pyplot
import os
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('project_contents/img/bgd.jpg')    

predictor_model = 'project_contents/shape_predictor_68_face_landmarks.dat'

# get facial features by using dlib
def get_points(image):
    
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    try:
        detected_face = face_detector(image, 1)[0]
    except:
        print('No face detected in image {}'.format(image))
    pose_landmarks = face_pose_predictor(image, detected_face)
    points = []
    for p in pose_landmarks.parts():
        points.append([p.x, p.y])

    
    x = image.shape[1] - 1
    y = image.shape[0] - 1
    points.append([0, 0])
    points.append([x // 2, 0])
    points.append([x, 0])
    points.append([x, y // 2])
    points.append([x, y])
    points.append([x // 2, y])
    points.append([0, y])
    points.append([0, y // 2])

    return np.array(points)


def get_triangles(points):
    
    return Delaunay(points).simplices


def affine_transform(input_image, input_triangle, output_triangle, size):
    
    warp_matrix = cv2.getAffineTransform(
        np.float32(input_triangle), np.float32(output_triangle))
    output_image = cv2.warpAffine(input_image, warp_matrix, (size[0], size[1]), None,
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return output_image


def morph_triangle(img1, img2, img, tri1, tri2, tri, alpha):
    
    rect1 = cv2.boundingRect(np.float32([tri1]))
    rect2 = cv2.boundingRect(np.float32([tri2]))
    rect = cv2.boundingRect(np.float32([tri]))

    tri_rect1 = []
    tri_rect2 = []
    tri_rect_warped = []

    for i in range(0, 3):
        tri_rect_warped.append(
            ((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
        tri_rect1.append(
            ((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
        tri_rect2.append(
            ((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))

    
    img1_rect = img1[rect1[1]:rect1[1] +
                     rect1[3], rect1[0]:rect1[0] + rect1[2]]
    img2_rect = img2[rect2[1]:rect2[1] +
                     rect2[3], rect2[0]:rect2[0] + rect2[2]]

    size = (rect[2], rect[3])
    warped_img1 = affine_transform(
        img1_rect, tri_rect1, tri_rect_warped, size)
    warped_img2 = affine_transform(
        img2_rect, tri_rect2, tri_rect_warped, size)

    
    img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

    
    mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

    
    img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
            rect[2]] * (1 - mask) + img_rect * mask


def morph_faces(filename1, filename2, alpha=0.5):
    
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    points1 = get_points(img1)
    points2 = get_points(img2)
    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    triangles = get_triangles(points)
    for i in triangles:
        x = i[0]
        y = i[1]
        z = i[2]

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]
        morph_triangle(img1, img2, img_morphed, tri1, tri2, tri, alpha)

    return np.uint8(img_morphed)

def read_image(file):
    return cv2.imread(file) 

def crop(image1, image2):
    w1, h1 = image1.shape[0], image1.shape[1]
    w2, h2 = image2.shape[0], image2.shape[1]
    print('image1 shape', image1.shape)
    print('image2 shape', image2.shape)
    wmin=min(w1, w2)
    hmin=min(h1, h2)
    print('minimum',wmin, hmin)
    
    x1_diff=(w1-wmin)//2
    x2_diff=(w2-wmin)//2
    y1_diff=(h1-hmin)//2
    y2_diff=(h2-hmin)//2
    print('diff image1', x1_diff, y1_diff)
    print('diff image2', x2_diff, y2_diff)


    image1 = image1[x1_diff:w1-x1_diff, y1_diff:h1-y1_diff]
    image2 = image2[x2_diff:w2-x2_diff, y2_diff:h2-y2_diff]
    return image1, image2

def save_image(img, path):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # im = Image.fromarray((img * 1).astype(np.uint8)).convert('RGB')
    im = Image.fromarray((img * 255).astype(np.uint8))
    im.save(f"{path}.jpeg")

def convert_to_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # im = Image.fromarray((img * 255).astype(np.uint8)).convert('RGB')
    im = Image.fromarray((img * 1).astype(np.uint8)).convert('RGB')
    return im

def main(img1, img2, alpha):
    points1 = get_points(img1)
    points2 = get_points(img2)
    points = (1 - alpha) * np.array(points1) + alpha * np.array(points2)
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)
    triangles1 = get_triangles(points1)
    triangles2 = get_triangles(points2)
    triangles = get_triangles(points)
    for i in triangles:
    
        # Calculate the frame of triangles
        x = i[0]
        y = i[1]
        z = i[2]

        tri1 = [points1[x], points1[y], points1[z]]
        tri2 = [points2[x], points2[y], points2[z]]
        tri = [points[x], points[y], points[z]]
        
        rect1 = cv2.boundingRect(np.float32([tri1]))
        rect2 = cv2.boundingRect(np.float32([tri2]))
        rect = cv2.boundingRect(np.float32([tri]))

        tri_rect1 = []
        tri_rect2 = []
        tri_rect_warped = []
        
        for i in range(0, 3):
            tri_rect_warped.append(((tri[i][0] - rect[0]), (tri[i][1] - rect[1])))
            tri_rect1.append(((tri1[i][0] - rect1[0]), (tri1[i][1] - rect1[1])))
            tri_rect2.append(((tri2[i][0] - rect2[0]), (tri2[i][1] - rect2[1])))
        
        # Accomplish the affine transform in triangles
        img1_rect = img1[rect1[1]:rect1[1] + rect1[3], rect1[0]:rect1[0] + rect1[2]]
        img2_rect = img2[rect2[1]:rect2[1] + rect2[3], rect2[0]:rect2[0] + rect2[2]]

        size = (rect[2], rect[3])
        warped_img1 = affine_transform(img1_rect, tri_rect1, tri_rect_warped, size)
        warped_img2 = affine_transform(img2_rect, tri_rect2, tri_rect_warped, size)
        
        # Calculate the result based on alpha
        img_rect = (1.0 - alpha) * warped_img1 + alpha * warped_img2

        # Generate the mask
        mask = np.zeros((rect[3], rect[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(tri_rect_warped), (1.0, 1.0, 1.0), 16, 0)

        # Accomplish the mask in the merged image
        img_morphed[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = \
            img_morphed[rect[1]:rect[1] + rect[3], rect[0]:rect[0] +
                rect[2]] * (1 - mask) + img_rect * mask 
    return img_morphed 

def printing(img):  
    st.write(img.shape)
    st.write(img[1,1,:])

templates = ['project_contents/img/style0.jpeg', 'project_contents/img/style2.jpeg', 'project_contents/img/style10.jpg']


def upload_template(option):
    # rand = np.random.randint(len(templates), size=1)
    scary_image = templates[option-1]
    img2 = read_image(scary_image)  
    return img2

st.markdown("<h1 style='text-align: center; color: white;'>Face your Fear!</h1>", unsafe_allow_html=True)

face_image= st.file_uploader("Upload a selfie with your face centred in the middle of the photo...")  #image 1 
alpha = st.slider('Scary Level', 0.0, 1.0, 0.1)

st.image([read_image(templates[0]), read_image(templates[1]), read_image(templates[2])], width=210)
option = st.selectbox(
     'Choose your after life?',
     (1, 2, 3))

st.write('You selected:', option)

# face_image = 'img/input.jpg'


if face_image is not None:
    img1 = Image.open(face_image)
    img1 =np.asarray(img1)
    img1 = img1[:,:,:3]
    img2 = upload_template(option)
    # print('Before crop')
   
    temp = np.copy(img1)
    blue, green, red = temp[:,:,0], temp[:,:,1], temp[:,:,2]

    img1[:,:,0] = red
    img1[:,:,2] = blue
    


    img1, img2 = crop(img1, img2) 
    # print('After crop')
    # printing(img1)
    # printing(img2)
    img3 = main(img1, img2, alpha)
    # print('final image')
    # printing(img2)
    img3 = convert_to_img(img3)
    
    st.image(img3)
    # save_image(img3, output)
