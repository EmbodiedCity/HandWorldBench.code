from pathlib import Path
from region_observations import clean_entities

def selector_prompt():
    return 'From these REFERENCE images, list only the manipulated objects, tools and operated targets/parts. Exclude hands, decoration, unused objects and background furniture. Describe each requested visible region by appearance and location so it can be segmented later. Do not judge task success, identity preservation or physical state. Return ONLY JSON {"objects":[{"description":string,"role":string}],"uncertainties":[string]}. At most 6 regions. Do not invent hidden objects.'

def select_objects(observer, frames, folder, *, separate_instances=False):
    from PIL import Image
    indices = sorted(set((round(i * (len(frames) - 1) / 7) for i in range(8))))
    images = []
    for i in indices:
        with Image.open(frames[i]['path']) as im:
            images.append(im.convert('RGB').copy())
    prompt = selector_prompt()
    if separate_instances:
        prompt += ' Describe ONE physical object instance per entry, never a pair or collection. Separate identical objects using appearance and their initial location in the first reference image. Keep each identity fixed as it moves; do not create entries for successive positions of the same object.'
    answer = observer.ask(images, prompt, Path(folder) / 'raw/object_selection.json', [f"REFERENCE time {frames[i]['timestamp_seconds']:.6f} s" for i in indices], 1600)
    objects = answer.get('objects')
    if not isinstance(objects, list) or len(objects) > 6:
        raise ValueError('Reference selector must return at most six explicit object regions')
    entities = []
    for i, item in enumerate(objects):
        if not isinstance(item, dict) or not isinstance(item.get('description'), str) or (not item['description'].strip()):
            raise ValueError('Invalid selected object description')
        entities.append(dict(id=f'object_{i + 1}', type='object', description=item['description'], role=item.get('role'), weight=1.0))
    return (clean_entities(entities) if entities else [], {'requests': 1, 'sample_indices': indices, 'source': 'reference_pixels_only', 'uncertainties': answer.get('uncertainties', [])})


def select_and_localize(observer, frames, folder):
    """Resolve reference identity conflicts once, using reference pixels only."""
    from region_observations import locate_objects, LocalizationIdentityError
    from handworldbench.runtime.storage import write
    folder = Path(folder)
    entities, selection = select_objects(observer, frames, folder / 'selection')
    try:
        prompted, _, audit = locate_objects(observer, frames, entities, folder / 'localization', stride=4, workers=1, role='reference')
    except LocalizationIdentityError as exc:
        write(folder / 'identity_recovery.json', dict(reason=str(exc), original_entities=entities, status='retrying', max_corrections=1))
        entities, corrected = select_objects(observer, frames, folder / 'identity_retry/selection', separate_instances=True)
        prompted, _, audit = locate_objects(observer, frames, entities, folder / 'identity_retry/localization', stride=4, workers=1, role='reference')
        selection = dict(corrected, requests=selection['requests'] + corrected['requests'], identity_corrections=1)
        write(folder / 'identity_recovery.json', dict(reason=str(exc), corrected_entities=entities, status='resolved', max_corrections=1))
    return entities, selection, prompted, audit
