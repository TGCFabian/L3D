import cv2


class LedResults:

    def __init__(self, center, contours):
        self.center = center
        self.contours = contours

class LedFinder:

    def __init__(self, threshold):
        self.threshold = threshold

    def find_led(self, image):

        _, image_thresh = cv2.threshold(image, self.threshold, 255, cv2.THRESH_TOZERO)

        contours, _ = cv2.findContours(image_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        moments = cv2.moments(image_thresh)

        if moments["m00"] == 0:
            return None

        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])

        return LedResults((center_x, center_y), contours)

    @staticmethod
    def draw_results(image, results):

        render_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if results:
            cv2.drawContours(render_image, results.contours, -1, (255, 0, 0), 1)
            cv2.drawMarker(render_image, results.center, (0, 255, 0), markerSize=100)

        return render_image