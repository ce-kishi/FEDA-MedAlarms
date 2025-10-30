import sed_eval
import dcase_util
import csv
import os
import re
import numpy as np

result_dir = 'result_fold/'
result_files = os.listdir(result_dir)
result_groups = {}
for file in result_files:
    if file.endswith('.txt'):
        group_key = re.sub(r'_fold\d', '', file)
        if group_key not in result_groups:
            result_groups[group_key] = []
        result_groups[group_key].append(os.path.join(result_dir, file))

print("グループ別評価結果 (平均 ± 標準偏差)")
print("Group\tSegment-based F1\tSegment-based Precision\tSegment-based Recall\tSegment-based ER\tEvent-based F1\tEvent-based Precision\tEvent-based Recall\tEvent-based ER")


# 各グループ（実験条件）ごとに評価を実行
for group_name, file_paths in result_groups.items():
    print(f"\n処理中: {group_name} ({len(file_paths)} folds)")

    # 各foldの評価結果を格納するリスト
    fold_results = {
        'segment_f1': [],
        'segment_precision': [],
        'segment_recall': [],
        'segment_er': [],
        'event_f1': [],
        'event_precision': [],
        'event_recall': [],
        'event_er': [],
        'segment_class_wise': [],
        'event_class_wise': []
    }

    # 各foldのファイルを評価
    for fold_file in file_paths:
        file_list = [
            {
                'reference_file': 'eval_reference.txt',
                'estimated_file': fold_file
            }
        ]

        data = []

        # Get used event labels
        all_data = dcase_util.containers.MetaDataContainer()
        for file_pair in file_list:
            reference_event_list = sed_eval.io.load_event_list(
                filename=file_pair['reference_file']
            )
            estimated_event_list = sed_eval.io.load_event_list(
                filename=file_pair['estimated_file']
            )

            data.append({'reference_event_list': reference_event_list,
                         'estimated_event_list': estimated_event_list})

            all_data += reference_event_list

        event_labels = all_data.unique_event_labels

        time_res = 0.1  # 指定した時間で平均化して評価。
        t_col = 2.0  # リファレンスと結果の開始位置・終了位置の時間差をいくら許容するか。

        # Start evaluating
        # Create metrics classes, define parameters
        segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
            event_label_list=event_labels,
            time_resolution=time_res
        )

        event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
            event_label_list=event_labels,
            t_collar=t_col
        )

        # Go through files
        for file_pair in data:
            segment_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )

            event_based_metrics.evaluate(
                reference_event_list=file_pair['reference_event_list'],
                estimated_event_list=file_pair['estimated_event_list']
            )

        # Get metrics for this fold
        overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
        overall_event_based_metrics = event_based_metrics.results_overall_metrics()

        # Get class-wise metrics for this fold
        class_wise_segment_metrics = segment_based_metrics.results_class_wise_metrics()
        class_wise_event_metrics = event_based_metrics.results_class_wise_metrics()

        # 各foldの結果を格納
        fold_results['segment_f1'].append(
            overall_segment_based_metrics['f_measure']['f_measure'])
        fold_results['segment_precision'].append(
            overall_segment_based_metrics['f_measure']['precision'])
        fold_results['segment_recall'].append(
            overall_segment_based_metrics['f_measure']['recall'])
        fold_results['segment_er'].append(
            overall_segment_based_metrics['error_rate']['error_rate'])
        fold_results['event_f1'].append(
            overall_event_based_metrics['f_measure']['f_measure'])
        fold_results['event_precision'].append(
            overall_event_based_metrics['f_measure']['precision'])
        fold_results['event_recall'].append(
            overall_event_based_metrics['f_measure']['recall'])
        fold_results['event_er'].append(
            overall_event_based_metrics['error_rate']['error_rate'])

        # クラス別メトリクスを格納
        fold_results['segment_class_wise'].append(class_wise_segment_metrics)
        fold_results['event_class_wise'].append(class_wise_event_metrics)

    # 平均と標準偏差を計算
    segment_f1_mean = np.mean(fold_results['segment_f1'])
    segment_f1_std = np.std(fold_results['segment_f1'])
    segment_precision_mean = np.mean(fold_results['segment_precision'])
    segment_precision_std = np.std(fold_results['segment_precision'])
    segment_recall_mean = np.mean(fold_results['segment_recall'])
    segment_recall_std = np.std(fold_results['segment_recall'])
    segment_er_mean = np.mean(fold_results['segment_er'])
    segment_er_std = np.std(fold_results['segment_er'])

    event_f1_mean = np.mean(fold_results['event_f1'])
    event_f1_std = np.std(fold_results['event_f1'])
    event_precision_mean = np.mean(fold_results['event_precision'])
    event_precision_std = np.std(fold_results['event_precision'])
    event_recall_mean = np.mean(fold_results['event_recall'])
    event_recall_std = np.std(fold_results['event_recall'])
    event_er_mean = np.mean(fold_results['event_er'])
    event_er_std = np.std(fold_results['event_er'])

    # クラス別メトリクスの統計計算
    all_classes = list(event_labels)
    class_wise_stats = {}

    for class_name in all_classes:
        class_wise_stats[class_name] = {
            'segment_f1': [],
            'segment_precision': [],
            'segment_recall': [],
            'segment_er': [],
            'event_f1': [],
            'event_precision': [],
            'event_recall': [],
            'event_er': []
        }

        # 各foldからクラス別メトリクスを収集
        for fold_idx in range(len(fold_results['segment_class_wise'])):
            seg_metrics = fold_results['segment_class_wise'][fold_idx]
            evt_metrics = fold_results['event_class_wise'][fold_idx]

            if class_name in seg_metrics:
                class_wise_stats[class_name]['segment_f1'].append(
                    seg_metrics[class_name]['f_measure']['f_measure'])
                class_wise_stats[class_name]['segment_precision'].append(
                    seg_metrics[class_name]['f_measure']['precision'])
                class_wise_stats[class_name]['segment_recall'].append(
                    seg_metrics[class_name]['f_measure']['recall'])
                class_wise_stats[class_name]['segment_er'].append(
                    seg_metrics[class_name]['error_rate']['error_rate'])

            if class_name in evt_metrics:
                class_wise_stats[class_name]['event_f1'].append(
                    evt_metrics[class_name]['f_measure']['f_measure'])
                class_wise_stats[class_name]['event_precision'].append(
                    evt_metrics[class_name]['f_measure']['precision'])
                class_wise_stats[class_name]['event_recall'].append(
                    evt_metrics[class_name]['f_measure']['recall'])
                class_wise_stats[class_name]['event_er'].append(
                    evt_metrics[class_name]['error_rate']['error_rate'])

    # グループ専用のCSVデータを作成（直接出力）
    group_csv_data = []

    # Overall metricsを追加
    group_csv_data.append({
        'Metric_Type': 'Segment-based',
        'Class': 'Overall',
        'F1_mean': segment_f1_mean,
        'F1_std': segment_f1_std,
        'Precision_mean': segment_precision_mean,
        'Precision_std': segment_precision_std,
        'Recall_mean': segment_recall_mean,
        'Recall_std': segment_recall_std,
        'ER_mean': segment_er_mean,
        'ER_std': segment_er_std
    })

    group_csv_data.append({
        'Metric_Type': 'Event-based',
        'Class': 'Overall',
        'F1_mean': event_f1_mean,
        'F1_std': event_f1_std,
        'Precision_mean': event_precision_mean,
        'Precision_std': event_precision_std,
        'Recall_mean': event_recall_mean,
        'Recall_std': event_recall_std,
        'ER_mean': event_er_mean,
        'ER_std': event_er_std
    })

    # Class-wise metricsを追加（計算済みのclass_wise_statsから直接追加）
    for class_name in all_classes:
        if class_wise_stats[class_name]['segment_f1']:
            seg_f1_mean = np.mean(class_wise_stats[class_name]['segment_f1'])
            seg_f1_std = np.std(class_wise_stats[class_name]['segment_f1'])
            seg_prec_mean = np.mean(
                class_wise_stats[class_name]['segment_precision'])
            seg_prec_std = np.std(
                class_wise_stats[class_name]['segment_precision'])
            seg_rec_mean = np.mean(
                class_wise_stats[class_name]['segment_recall'])
            seg_rec_std = np.std(
                class_wise_stats[class_name]['segment_recall'])
            seg_er_mean = np.mean(class_wise_stats[class_name]['segment_er'])
            seg_er_std = np.std(class_wise_stats[class_name]['segment_er'])

            group_csv_data.append({
                'Metric_Type': 'Segment-based',
                'Class': class_name,
                'F1_mean': seg_f1_mean,
                'F1_std': seg_f1_std,
                'Precision_mean': seg_prec_mean,
                'Precision_std': seg_prec_std,
                'Recall_mean': seg_rec_mean,
                'Recall_std': seg_rec_std,
                'ER_mean': seg_er_mean,
                'ER_std': seg_er_std
            })

        if class_wise_stats[class_name]['event_f1']:
            evt_f1_mean = np.mean(class_wise_stats[class_name]['event_f1'])
            evt_f1_std = np.std(class_wise_stats[class_name]['event_f1'])
            evt_prec_mean = np.mean(
                class_wise_stats[class_name]['event_precision'])
            evt_prec_std = np.std(
                class_wise_stats[class_name]['event_precision'])
            evt_rec_mean = np.mean(
                class_wise_stats[class_name]['event_recall'])
            evt_rec_std = np.std(class_wise_stats[class_name]['event_recall'])
            evt_er_mean = np.mean(class_wise_stats[class_name]['event_er'])
            evt_er_std = np.std(class_wise_stats[class_name]['event_er'])

            group_csv_data.append({
                'Metric_Type': 'Event-based',
                'Class': class_name,
                'F1_mean': evt_f1_mean,
                'F1_std': evt_f1_std,
                'Precision_mean': evt_prec_mean,
                'Precision_std': evt_prec_std,
                'Recall_mean': evt_rec_mean,
                'Recall_std': evt_rec_std,
                'ER_mean': evt_er_mean,
                'ER_std': evt_er_std
            })

    # グループ別CSVファイル出力（各グループの処理中に実行）
    csv_filename = f'./eval_result_fold/eval_meansd_t_res{time_res}_t_col{t_col}_{group_name}.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Metric_Type', 'Class', 'F1_mean', 'F1_std',
                      'Precision_mean', 'Precision_std', 'Recall_mean',
                      'Recall_std', 'ER_mean', 'ER_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(group_csv_data)

    print(f"グループCSV保存完了: {csv_filename}")

    # 結果を出力
    # 結果を出力
    print(f"{group_name}")
    print(f"Segment-based: F1={segment_f1_mean:.4f}±{segment_f1_std:.4f}, "
          f"Precision={segment_precision_mean:.4f}±{segment_precision_std:.4f}")
    print(f"Event-based: F1={event_f1_mean:.4f}±{event_f1_std:.4f}, "
          f"Precision={event_precision_mean:.4f}±{event_precision_std:.4f}")


print("すべてのグループの処理が完了しました。")
