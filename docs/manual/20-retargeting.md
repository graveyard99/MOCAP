# Retargeting

The fitted character is a metric intermediate, not the only possible production rig. It preserves stable names, hierarchy, rest proportions, skin weights, local rotations and global translation.

Export FBX for a skinned character, or BVH for skeleton-only interchange. In the destination DCC, establish a joint-name mapping, align rest poses and axes, and distinguish global root motion from local pelvis rotation. Verify limb lengths before introducing scale compensation.

A different digital double or custom skeleton generally needs a DCC-specific retarget solver. This application includes explicit naming/mapping facilities in the core, but no complete artist-facing MetaHuman/custom-rig retarget wizard.

Keep fitted actor proportions and your production character proportions separate. Retargeting changes the target rig's motion interpretation; it must not alter the measured camera solution or metric source trajectory. Record any intentional character-scale transform in the receiving scene.

Validate at least a neutral pose, bent-knee pose, reaching pose and root translation. Inspect planted feet after mapping: joint-axis/rest-pose differences can introduce foot sliding even when the source contact solve is sound.

Blender round-trip validation verifies this application's exported character. It does not certify every third-party retargeting/import plug-in. Maya, Houdini, Cinema 4D and Unreal integration needs facility testing with the intended target assets.
