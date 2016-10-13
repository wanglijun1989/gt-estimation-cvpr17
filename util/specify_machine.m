switch machine_id
    case 'local'
        machine_path = 'cvpr17-ILT-pretrain-fs';
    otherwise
        machine_path = machine_id;
end