import boto3


class FaceDetection:
    """
    This class will recognize face using aws service
    """

    def __init__(self, file_path, access_key, access_secret, access_token):
        """

        :param file_path:
        """
        self.photo = file_path
        self.aws_access_key_id = access_key
        self.aws_access_secret = access_secret
        self.access_token = access_token

        # initialize boto3 object
        self.client = boto3.client("rekognition", aws_access_key_id=self.aws_access_key_id,
                                   aws_secret_access_key=self.aws_access_secret,
                                   aws_session_token=self.access_token
                                   )

    def face_detection(self):
        """

        :return:
        """
        # read the image file passed
        with open(self.photo, 'rb') as img:
            response = self.client.detect_faces(Image={
                'Bytes': img.read()
            },
                Attributes=['ALL']
            )

        # parse the result to get the required fields
        if response['FaceDetails']:
            result = {
                'gender': response['FaceDetails'][0]['Gender']['Value'],
                'age_range': str(response['FaceDetails'][0]['AgeRange']['Low']) + '-' + str(
                    response['FaceDetails'][0]['AgeRange']['High']),
                'emotions': response['FaceDetails'][0]['Emotions'][0]['Type'],
                'sun_glasses': response['FaceDetails'][0]['Sunglasses']['Value'],
                'beard': response['FaceDetails'][0]['Beard']['Value'],
                'mustache': response['FaceDetails'][0]['Mustache']['Value'],
                'eyes_open': response['FaceDetails'][0]['EyesOpen']['Value'],
                'mouth_open': response['FaceDetails'][0]['MouthOpen']['Value'],
                'smile': response['FaceDetails'][0]['Smile']['Value']
            }
        else:
            result = 'Face could not be detected'
        return result

    def run(self):
        details = self.face_detection()

        # display the output
        if isinstance(details, dict):
            print('**************************The candidate has the following details*********************************')
            for each_key in details.keys():
                print(each_key + '=' + str(details[each_key]))

        else:
            print(details)


if __name__ == '__main__':
    # Initializing class and creating a class object
    fd = FaceDetection(file_path='your image path',
                       access_key='your access key',
                       access_secret='access secret',
                       access_token='token')
    fd.run()
