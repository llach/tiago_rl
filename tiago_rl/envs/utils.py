import pybullet as p


def link_to_idx(body_id):
    d = {p.getBodyInfo(body_id)[0].decode('UTF-8'): -1, }

    for _id in range(p.getNumJoints(body_id)):
        _name = p.getJointInfo(body_id, _id)[12].decode('UTF-8')
        d[_name] = _id

    return d


def joint_to_idx(body_id):
    return {key.decode(): value for (value, key) in [p.getJointInfo(body_id, i)[:2] for i in range(p.getNumJoints(body_id))]}
