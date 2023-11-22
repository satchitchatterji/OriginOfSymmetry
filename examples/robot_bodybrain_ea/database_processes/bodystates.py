from revolve2.modular_robot._body_state import BodyState
from pyrr import Quaternion, Vector3


def get_body_states_from_str(single_robot_body_state_str):
    # generate body states from the string of form:
    try:
        body_states = eval(single_robot_body_state_str) # eval() converts string to list of BodyState objects
    except Exception as e:
        print("Error: could not convert string to list of BodyState objects.")
        return None
    
    return body_states

if __name__ == "__main__":
    test_list = ['[BodyState(core_position=Vector3([0.        , 0.        , 0.09432938]), core_orientation=Quaternion([1., 0., 0., 0.])), BodyState(core_position=Vector3([-0.03402834,  0.00049798,  0.0694624 ]), core_orientation=Quaternion([ 0.98011376, -0.00213592, -0.19835909, -0.00511175]))]', '[BodyState(core_position=Vector3([0.        , 0.        , 0.03144313]), core_orientation=Quaternion([1., 0., 0., 0.])), BodyState(core_position=Vector3([0.28300224, 0.07629301, 0.03035856]), core_orientation=Quaternion([ 3.17403808e-01,  2.90003238e-03, -4.24113564e-04,\n             9.48285945e-01]))]', '[BodyState(core_position=Vector3([0.        , 0.        , 0.09432938]), core_orientation=Quaternion([1., 0., 0., 0.])), BodyState(core_position=Vector3([-0.0329439 , -0.00107748,  0.06977345]), core_orientation=Quaternion([ 0.98114091, -0.0017441 , -0.19328304, -0.00106623]))]', '[BodyState(core_position=Vector3([0.     , 0.     , 0.03015]), core_orientation=Quaternion([1., 0., 0., 0.])), BodyState(core_position=Vector3([-1.97522203e-19,  5.15127629e-16,  3.00422446e-02]), core_orientation=Quaternion([ 1.00000000e+00,  1.92686659e-17, -2.30988404e-20,\n             2.68824219e-20]))]', '[BodyState(core_position=Vector3([0.     , 0.     , 0.03015]), core_orientation=Quaternion([1., 0., 0., 0.])), BodyState(core_position=Vector3([-1.97522203e-19,  5.15127629e-16,  3.00422446e-02]), core_orientation=Quaternion([ 1.00000000e+00,  1.92686659e-17, -2.30988404e-20,\n             2.68824219e-20]))]']
    test_str = test_list[0]
    print(get_body_states_from_str(test_str))