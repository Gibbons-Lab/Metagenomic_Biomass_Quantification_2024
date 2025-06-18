#!/usr/bin/env nextflow
params.data_dir = "${launchDir}/data"
params.raw = "${params.data_dir }/fastq"
params.bowtie_index = "/proj/gibbons/refs/human_genome_ref/bt2_index/human_ref"
params.single_end=false

process removehost {
    cpus 1
    publishDir "${params.data_dir}/decontaminated"

    input:
    tuple val(id), path(reads)

    output:
    path("${id}_filtered_R*.fastq.gz")
    
    script:
    if (params.single_end)
        """
        hocort map bowtie2 -t 1 -x ${params.bowtie_index} -i ${reads[0]} -o "${id}_filtered_R1.fastq.gz" -t 1
        """

    else
        """
        hocort map bowtie2 -t 1 -x ${params.bowtie_index} -i ${reads[0]} ${reads[1]} -o "${id}_filtered_R1.fastq.gz" "${id}_filtered_R2.fastq.gz"
        """
}


workflow {
    // find files
    if (params.single_end) {
        Channel
            .fromPath("${params.raw}/*.fastq.gz")
            .map{row -> tuple(row.baseName.split("\\.fastq")[0], tuple(row))}
            .set{raw}
        n = file("${params.raw}/*.fastq.gz").size()
    } else {
        Channel
            .fromFilePairs([
                "${params.raw}/*_R{1,2}_001.fastq.gz",
                "${params.raw}/*_{1,2}.fastq.gz",
                "${params.raw}/*_R{1,2}.fastq.gz"
            ])
            .ifEmpty { error "Cannot find any read files in ${params.raw}!" }
            .set{raw}
        n = file("${params.raw}/*.f*.gz").size() / 2
    }
    removehost(raw)
}
