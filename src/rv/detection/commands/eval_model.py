from os.path import join, isfile
import json

import click

from rv.detection.commands.predict import _predict
from rv.detection.commands.eval_predictions import _eval_predictions
from rv.utils.files import (
    download_if_needed, get_local_path, upload_if_needed, make_dir)
from rv.utils.misc import load_projects
from rv.detection.commands.settings import default_channel_order, temp_root_dir


def save_eval_avgs(eval_paths, output_path):
    label_to_avgs = {}
    nb_projects = len(eval_paths)
    for eval_path in eval_paths:
        with open(eval_path, 'r') as eval_file:
            project_eval = json.load(eval_file)
            for label_eval in project_eval:
                label_name = label_eval['name']
                label_avgs = label_to_avgs.get(label_name, {})
                for key, val in label_eval.items():
                    if key == 'name':
                        label_avgs['name'] = label_name
                    else:
                        label_avgs[key] = \
                            label_avgs.get(key, 0) + (val / nb_projects)
                label_to_avgs[label_name] = label_avgs

    with open(output_path, 'w') as output_file:
        json.dump(list(label_to_avgs.values()), output_file, indent=4)


@click.command()
@click.argument('inference_graph_uri')
@click.argument('projects_uri')
@click.argument('label_map_uri')
@click.argument('output_uri')
@click.option('--use-cached-predictions', is_flag=True,
              default=False)
@click.option('--evals-uri', default=None)
@click.option('--predictions-uri', default=None)
@click.option('--channel-order', nargs=3, type=int,
              default=default_channel_order, help='Index of RGB channels')
@click.option('--chip-size', default=300)
@click.option('--score-thresh', default=0.5,
              help='Score threshold of predictions to keep')
@click.option('--merge-thresh', default=0.05,
              help='IOU threshold for merging predictions')
def eval_model(inference_graph_uri, projects_uri, label_map_uri, output_uri,
               use_cached_predictions, evals_uri, predictions_uri,
               channel_order, chip_size, score_thresh, merge_thresh):
    """Evaluate a model on a set of projects with ground truth annotations.

    Makes predictions using a model on a set of projects and then compares them
    with ground truth annotations, saving the average precision and recall
    across the projects.

    Args:
        inference_graph_uri: the inference graph of the model to evaluate
        projects_uri: the JSON file with the images and annotations for a
            set of projects
        label_map_uri: label map for the model
        output_uri: the destination for the JSON output
    """
    temp_dir = join(temp_root_dir, 'eval-model')
    make_dir(temp_dir, force_empty=True)

    if predictions_uri is None:
        predictions_uri = join(temp_dir, 'predictions')
    # TODO sync predictions uri with predictions dir
    predictions_dir = get_local_path(predictions_uri, temp_dir)
    make_dir(predictions_dir, check_empty=(not use_cached_predictions))

    if evals_uri is None:
        evals_uri = join(temp_dir, 'evals')
    evals_dir = get_local_path(evals_uri, temp_dir)
    make_dir(evals_dir, check_empty=True)

    projects_path = download_if_needed(projects_uri, temp_dir)
    project_ids, image_paths_list, annotations_paths = \
        load_projects(projects_path, temp_dir)

    output_path = get_local_path(output_uri, temp_dir)
    make_dir(output_path, use_dirname=True)

    # Run prediction and evaluation on each project.
    eval_paths = []
    for project_id, image_paths, annotations_path in \
            zip(project_ids, image_paths_list, annotations_paths):
        predictions_path = join(predictions_dir, '{}.json'.format(project_id))
        if use_cached_predictions:
            if not isfile(predictions_path):
                raise ValueError(
                    '--use_cached_predictions is set but {} is missing'.format(
                        predictions_path))
        else:
            print('Making predictions and storing in {}'.format(
                predictions_path))
            _predict(inference_graph_uri, label_map_uri, image_paths,
                     predictions_path, channel_order=channel_order,
                     chip_size=chip_size, score_thresh=score_thresh,
                     merge_thresh=merge_thresh)
        eval_path = join(evals_dir, '{}.json'.format(project_id))
        eval_paths.append(eval_path)
        _eval_predictions(
            image_paths, label_map_uri, annotations_path, predictions_path,
            eval_path)

    save_eval_avgs(eval_paths, output_path)
    upload_if_needed(output_path, output_uri)
    upload_if_needed(predictions_dir, predictions_uri)
    upload_if_needed(evals_dir, evals_uri)


if __name__ == '__main__':
    eval_model()
