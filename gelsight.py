import cv2
from gelsight_helper import GelSightMini

DEVICE_ID = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._GelSight_Mini_R0B_28F5-K4RR_28F5K4RR-video-index0"
img_w = 640 # Target FULLWIDTH
img_h = 480 # Target FULLHEIGHT
brd_frac_crop = 0.15 


def main():
    gelsight = GelSightMini(
        device_idx=DEVICE_ID,
        target_width=img_w,
        target_height=img_h,
        border_fraction=brd_frac_crop
    )

    while True:
        img = gelsight.update()

        if img.shape[0] > 0:
            imbgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame_rgb', imbgr)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
